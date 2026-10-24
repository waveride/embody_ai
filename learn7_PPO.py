import argparse
import time
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # 注册 ManiSkill 环境
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


MODEL_PATH = Path(__file__).resolve().parent / "ppo_pickcube"


def make_env(render=False):
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        sim_backend="cpu",
        robot_uids="panda",
        obs_mode="state",                  # 输入状态向量，不使用图像
        control_mode="pd_ee_delta_pos",    # 末端 xyz 增量 + 夹爪控制
        reward_mode="normalized_dense",    # 使用任务内置的稠密奖励
        render_mode="human" if render else None,
        max_episode_steps=100,
    )

    # 转换为标准 Gymnasium 接口：
    # 观测变为 numpy，奖励变为标量，移除环境批次维度
    env = CPUGymWrapper(env)

    # 记录每回合累计奖励与步数，供训练日志使用
    env = Monitor(env)
    return env


def train(total_steps):
    env = make_env(render=False)

    try:
        print("观测空间：", env.observation_space)
        print("动作空间：", env.action_space)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,       # 每采集约 2048 步，进行一次训练阶段
            batch_size=64,
            n_epochs=10,        # 每批采集的数据重复优化 10 轮
            gamma=0.99,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
            seed=0,
            device="cpu",
        )

        # 内部自动执行：采样动作 → 环境交互 → 收集奖励 → 更新网络
        model.learn(total_timesteps=total_steps)

        model.save(str(MODEL_PATH))
        print(f"模型已保存：{MODEL_PATH}.zip")

    finally:
        env.close()


def test(episodes=10):
    if not MODEL_PATH.with_suffix(".zip").exists():
        raise FileNotFoundError("尚未找到模型，请先运行 train。")

    env = make_env(render=True)
    model = PPO.load(str(MODEL_PATH), device="cpu")
    success_count = 0

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=100_000 + episode)

            episode_reward = 0.0
            success_once = False
            steps = 0

            while True:
                # 测试时仅预测动作，不更新网络
                action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                steps += 1
                episode_reward += float(reward)
                success_once |= bool(info.get("success", False))

                env.render()
                time.sleep(0.02)

                if terminated or truncated:
                    break

            success_count += int(success_once)
            result = "成功" if success_once else "失败"

            print(
                f"测试 {episode + 1}/{episodes} | {result} | "
                f"步数={steps} | 累计奖励={episode_reward:.2f}"
            )

        print(f"回合内曾成功比例：{success_count}/{episodes}")

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--steps", type=int, default=300_000)
    #args = parser.parse_args()
    args = parser.parse_args(["train", "--steps", "10000"])

    if args.mode == "train":
        train(args.steps)
    else:
        test()