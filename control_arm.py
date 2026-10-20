import time
import gymnasium as gym
import mani_skill.envs
import numpy as np


def main():
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        sim_backend="cpu",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="human",
    )

    try:
        obs, info = env.reset(seed=0)
        print("动作空间：", env.action_space)

        for step in range(40):
            # 前 20 步上移，后 20 步下移
            dz = 0.2 if step < 20 else -0.2

            # 动作：[x方向增量, y方向增量, z方向增量, 夹爪指令]
            action = np.array(
                [0.0, 0.0, dz, 1.0],
                dtype=np.float32,
            )

            # 匹配环境要求的动作形状
            action = action.reshape(env.action_space.shape)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.1)

            if step % 10 == 0:
                position = env.unwrapped.agent.tcp.pose.p
                print(f"第 {step} 步，末端位置：{position}")

            if terminated.item() or truncated.item():
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()