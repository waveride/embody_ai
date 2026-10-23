print("开始执行文件", __file__, flush=True)

import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MAX_STEPS = 400


def to_numpy(tensor):
    return tensor[0].detach().cpu().numpy().copy()


def make_env(render=False):
    return gym.make(
        "PickCube-v1",
        robot_uids="panda",
        num_envs=1,
        sim_backend="cpu",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="human" if render else None,
        max_episode_steps=MAX_STEPS,
    )


# ============================================================
# 1. 专家策略：你已经学过的状态机
# ============================================================

class Expert:
    def __init__(self, base):
        root_q = to_numpy(base.agent.robot.pose.q)

        if not np.allclose(
            np.abs(root_q), [1, 0, 0, 0], atol=1e-4
        ):
            raise RuntimeError("此专家要求机器人底座坐标轴与世界轴对齐。")

        cube = to_numpy(base.cube.pose.p)

        self.stage = "approach"
        self.target = cube + np.array([0, 0, 0.10])
        self.close_steps = 0
        self.initial_z = cube[2]

    def act(self, base):
        tcp = to_numpy(base.agent.tcp.pose.p)
        cube = to_numpy(base.cube.pose.p)
        goal = to_numpy(base.goal_site.pose.p)

        reached = np.linalg.norm(self.target - tcp) < 0.005

        # 根据执行结果更新阶段和目标
        if self.stage == "approach" and reached:
            self.stage = "descend"
            self.target = cube.copy()

        elif self.stage == "descend" and reached:
            self.stage = "close"
            self.target = tcp.copy()
            self.close_steps = 0

        elif self.stage == "close" and self.close_steps >= 25:
            self.stage = "lift"
            self.target = tcp + np.array([0, 0, 0.12])

        elif self.stage == "lift" and reached:
            lifted = cube[2] - self.initial_z > 0.05
            grasped = bool(base.agent.is_grasping(base.cube).item())

            if not (lifted and grasped):
                return None

            self.stage = "transport"

            # 目标要求方块中心到达 goal，因此补偿 TCP 与方块中心偏移
            tcp_to_cube_offset = tcp - cube
            self.target = goal + tcp_to_cube_offset

        # transport 阶段保持最终目标，等待环境判断成功
        gripper = (
            1.0 if self.stage in ("approach", "descend") else -1.0
        )

        if self.stage == "close":
            self.close_steps += 1

        error = self.target - tcp
        move = np.clip(6.0 * error, -0.3, 0.3)

        return np.concatenate(
            [move, [gripper]]
        ).astype(np.float32)


# ============================================================
# 2. 采集：记录 obs_t 和 action_t，只保留成功轨迹
# ============================================================

def collect(args):
    env = make_env(args.render)

    all_obs = []
    all_actions = []
    all_episode_ids = []

    successes = 0

    try:
        for attempt in range(args.max_attempts):
            obs, info = env.reset(seed=attempt)
            expert = Expert(env.unwrapped)

            episode_obs = []
            episode_actions = []
            success = False

            for step in range(MAX_STEPS):
                action = expert.act(env.unwrapped)

                if action is None:
                    break

                # 关键：保存动作执行前的观测
                episode_obs.append(to_numpy(obs))
                episode_actions.append(action.copy())

                obs, reward, terminated, truncated, info = env.step(
                    action.reshape(env.action_space.shape)
                )

                if args.render:
                    env.render()

                success = bool(info["success"].item())

                if success or terminated.item() or truncated.item():
                    break

            if success:
                all_obs.append(np.asarray(episode_obs, dtype=np.float32))
                all_actions.append(
                    np.asarray(episode_actions, dtype=np.float32)
                )
                all_episode_ids.append(
                    np.full(len(episode_obs), successes, dtype=np.int64)
                )
                successes += 1

            print(
                f"尝试 {attempt + 1:4d} | "
                f"成功轨迹 {successes}/{args.episodes} | "
                f"本轮 {'成功' if success else '失败'}"
            )

            if successes >= args.episodes:
                break

    finally:
        env.close()

    if successes < 2:
        raise RuntimeError(
            "成功轨迹少于 2 条。请先用 --render 检查专家抓取程序。"
        )

    output = Path(args.data)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        obs=np.concatenate(all_obs),
        actions=np.concatenate(all_actions),
        episode_ids=np.concatenate(all_episode_ids),
    )

    print(f"\n数据保存到：{output}")
    print(f"成功轨迹数：{successes}")
    print(f"观测—动作对数量：{sum(len(x) for x in all_obs)}")

    if successes < args.episodes:
        print("已达到最大尝试次数，保存了当前采集到的成功轨迹。")


# ============================================================
# 3. 策略网络：观测 → 4 维动作
# ============================================================

class Policy(nn.Module):
    def __init__(self, obs_dim, obs_mean, obs_std):
        super().__init__()

        # 标准化参数随模型一起保存
        self.register_buffer("obs_mean", obs_mean.clone())
        self.register_buffer("obs_std", obs_std.clone())

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Tanh(),
        )

    def forward(self, obs):
        obs = (obs - self.obs_mean) / self.obs_std
        return self.net(obs)


# ============================================================
# 4. 训练：用专家动作作为监督标签
# ============================================================

def train(args):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    with np.load(args.data) as data:
        observations = data["obs"].copy()
        actions = data["actions"].copy()
        episode_ids = data["episode_ids"].copy()

    # 按整条轨迹划分，避免同一轨迹的相邻帧进入不同集合
    unique_ids = np.unique(episode_ids)

    if len(unique_ids) < 2:
        raise RuntimeError("至少需要两条轨迹。")

    rng.shuffle(unique_ids)
    n_val = max(1, int(len(unique_ids) * 0.2))
    val_ids = unique_ids[:n_val]

    val_mask = np.isin(episode_ids, val_ids)
    train_mask = ~val_mask

    x_train = torch.from_numpy(observations[train_mask]).float()
    y_train = torch.from_numpy(actions[train_mask]).float()
    x_val = torch.from_numpy(observations[val_mask]).float()
    y_val = torch.from_numpy(actions[val_mask]).float()

    # 只用训练集计算标准化参数
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-3)

    model = Policy(x_train.shape[1], mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=256,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=512,
    )

    checkpoint_path = Path(args.model)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    print(f"观测维度：{x_train.shape[1]}")
    print(f"训练帧数：{len(x_train)}，验证帧数：{len(x_val)}")

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0

        for obs_batch, action_batch in train_loader:
            predicted = model(obs_batch)
            loss = nn.functional.mse_loss(predicted, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * len(obs_batch)

        model.eval()
        val_total = 0.0

        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                predicted = model(obs_batch)
                loss = nn.functional.mse_loss(predicted, action_batch)
                val_total += loss.item() * len(obs_batch)

        train_loss = train_total / len(x_train)
        val_loss = val_total / len(x_val)

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                {
                    "obs_dim": x_train.shape[1],
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d} | "
                f"train MSE={train_loss:.6f} | "
                f"val MSE={val_loss:.6f}"
            )

    print(f"\n最优模型：{checkpoint_path}")
    print(f"最优验证 MSE：{best_val:.6f}")


# ============================================================
# 5. 测试：只使用网络，不调用专家状态机
# ============================================================

def evaluate(args):
    checkpoint = torch.load(
        args.model,
        map_location="cpu",
        weights_only=True,
    )

    obs_dim = checkpoint["obs_dim"]
    model = Policy(
        obs_dim,
        torch.zeros(obs_dim),
        torch.ones(obs_dim),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    env = make_env(args.render)
    successes = 0

    try:
        for episode in range(args.episodes):
            # 与采集阶段使用不同的随机种子
            seed = args.test_seed + episode
            obs, info = env.reset(seed=seed)
            success = False

            if obs.shape[-1] != obs_dim:
                raise RuntimeError("观测维度与训练时不一致。")

            for step in range(MAX_STEPS):
                obs_tensor = torch.from_numpy(
                    to_numpy(obs)
                ).float().unsqueeze(0)

                with torch.no_grad():
                    action = model(obs_tensor)[0].numpy()

                # 与示范使用同样的移动指令上限
                action[:3] = np.clip(action[:3], -0.3, 0.3)
                action[3] = np.clip(action[3], -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(
                    action.reshape(env.action_space.shape)
                )

                if args.render:
                    env.render()

                success = bool(info["success"].item())

                if success or terminated.item() or truncated.item():
                    break

            successes += int(success)

            print(
                f"测试 {episode + 1}/{args.episodes} | "
                f"seed={seed} | "
                f"{'成功' if success else '失败'} | "
                f"步数={step + 1}"
            )

    finally:
        env.close()

    print(
        f"\n测试成功率：{successes}/{args.episodes} "
        f"= {successes / args.episodes:.1%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "train", "eval"])
    parser.add_argument("--data", default="bc_data.npz")
    parser.add_argument("--model", default="bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=100000)
    parser.add_argument("--render", action="store_true")
    #args = parser.parse_args()
#     args = parser.parse_args([
#     "collect",
#     "--episodes", "100",
#     "--max-attempts", "1000",
# ])
#     args = parser.parse_args([
#     "train",
#     "--epochs", "100",
# ])
    args = parser.parse_args([
    "eval",
    "--episodes", "10",
    "--render",
])

    if args.episodes < 1:
        parser.error("--episodes 必须大于 0")

    if args.mode == "collect":
        collect(args)
    elif args.mode == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    print("准备进入 main()", flush=True)
    main()