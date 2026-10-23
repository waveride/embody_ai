import time

import gymnasium as gym
import mani_skill.envs
import numpy as np


SEED = 2
GAIN = 6.0
MAX_ACTION = 0.3
TOLERANCE = 0.005

LIFT_HEIGHT = 0.12
MOVE_OFFSET = np.array([0.10, 0.0, 0.0], dtype=np.float32)

CLOSE_STEPS = 20
RELEASE_STEPS = 20
STAGE_TIMEOUT = 150
MAX_STEPS = 1000


def test():


# 物体中心在世界坐标系中的位置
    object_position = np.array([0.3, 0.0, 0.02])

# 物体绕世界 z 轴旋转 30°
    theta = np.deg2rad(30)

    rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta),  np.cos(theta), 0.0],
    [0.0,            0.0,           1.0],
    ])

# 假设物体自身 x 轴沿长边：
# 沿长边偏移 3 厘米，再沿物体 z 轴向上偏移 10 厘米
    offset_local = np.array([0.03, 0.0, 0.10])

    offset_world = rotation @ offset_local
    target_world = object_position + offset_world

    print("局部偏移：", offset_local)
    print("世界偏移：", offset_world)
    print("目标位置：", target_world)

def test1():


# 物体中心的世界坐标
    object_position = np.array([0.30, 0.00, 0.02])

# 物体绕世界 z 轴旋转 30°
    theta = np.deg2rad(30)

    object_rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta),  np.cos(theta), 0.0],
    [0.0,            0.0,           1.0],
])

# ---------- 计算位置 ----------

# 接近点：物体中心正上方 10 厘米
    offset_local = np.array([0.0, 0.0, 0.10])

    approach_position = (
    object_rotation @ offset_local + object_position
)

# ---------- 计算朝向 ----------

# 简化抓取坐标系：
# x 沿物体长边，y 沿短边反方向，z 朝下
    grasp_rotation_local = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])

    grasp_rotation_world = (
    object_rotation @ grasp_rotation_local
)

    print("接近点的世界坐标：")
    print(np.round(approach_position, 3))

    print("\n抓取坐标系的世界朝向：")
    print(np.round(grasp_rotation_world, 3))

    print("\n抓取坐标系 x 轴的世界方向：")
    print(np.round(grasp_rotation_world[:, 0], 3))

    print("\n抓取坐标系 z 轴的世界方向：")
    print(np.round(grasp_rotation_world[:, 2], 3))

def to_numpy(tensor):
    return tensor[0].detach().cpu().numpy().copy()


def check_base_rotation(base):
    root_q = to_numpy(base.agent.robot.pose.q)
    if not np.allclose(
        np.abs(root_q),
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-4,
    ):
        raise RuntimeError("底座存在旋转，需要先转换控制误差的坐标系。")


def main():
    env = gym.make(
        "PickCube-v1",
        robot_uids="panda",
        num_envs=1,
        sim_backend="cpu",
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="human",
        max_episode_steps=MAX_STEPS,
    )

    try:
        obs, info = env.reset(seed=SEED)
        base = env.unwrapped
        check_base_rotation(base)



    finally:
        env.close()


if __name__ == "__main__":
    test1()