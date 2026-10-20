# 基于位置反馈的机械臂接近控制


import time

import gymnasium as gym
import mani_skill.envs
import numpy as np

Seed = 2
gain = 6.0
max_action = 0.3
tolerance = 0.001
height_offset = 0.01

def to_numpy(tensor):
#ManiSkill 返回的位置通常是 PyTorch Tensor。这个函数把它转换为方便计算的 NumPy 数组。
    return tensor[0].detach().cpu().numpy().copy()

def main():
    env = gym.make(
        "PickCube-v1",             #选择抓取方块任务，官方定义
        robot_uids = "panda",       #使用 Panda 机械臂
        num_envs = 1,                  #并行环境数量：1 表示只运行 1 个仿真场景；ManiSkill 原生支持向量化并行，可同时运行成百上千个环境加速数据收集。
        sim_backend = "cpu",         
        obs_mode = "state",      #`state` 表示返回完整状态向量（机械臂关节角、立方体位姿、目标位置等全部真实信息），属于完美观测；可选 `rgb`（图像）、`pointcloud`（点云）等视觉观测模式。
        control_mode = "pd_ee_delta_pos",   #e：end-effector，末端执行器。delta_pos：位置增量。pd：底层使用 PD 控制。给出末端移动指令，控制器通过逆运动学计算关节目标，再驱动。因此动作前 3 维是末端 x/y/z 位移增量，第 4 维是夹爪开度指令。
        render_mode = "human",         #打开可视化窗口，可选 `rgb_array`（返回图像数组）、`None`（后台静默运行）。
        max_episode_steps=200
    )

    try:
        obs, info = env.reset(seed=Seed)   # 开始一轮任务，将机器人和方块放到初始状态。随机种子一致保持复现,obs：初始观测。info：附加信息。
        base = env.unwrapped               # 取得内部原始任务对象。这里把它命名为 base，方便访问：

        # 检查旋转，如果旋转了需要对齐坐标系
        root_q = to_numpy(base.agent.robot.pose.q)  # .p：位置。.q：用四元数表示的方向。

        if not np.allclose(
        np.abs(root_q), [1.0, 0.0, 0.0, 0.0], atol=1e-4     #判断两个数组是否在容差内接近， 等价于有没有旋转
        ):
            raise RuntimeError("底座存在旋转，需要先转换控制误差的坐标系。")

        tcp_pose = to_numpy(base.agent.tcp.pose.p)  # 末端执行器位置,世界坐标[x,y,z]，TCP 是夹爪的控制参考点，可以理解为两根夹指之间的一个点。
        cube_pose = to_numpy(base.cube.pose.p)  # 方块位置， 世界坐标

        target_pose = target_pos = cube_pose + np.array([0.0, 0.0, height_offset], dtype=np.float32)
        # 设置目标点，目标点与方块的 x、y 坐标相同，z 坐标比方块中心高 12 cm。

        print("观测形状：", obs.shape)
        print("动作空间：", env.action_space)
        print("初始 TCP 位置：", tcp_pose)
        print("方块中心位置：", cube_pose)
        print("今天的接近目标：", target_pose)

        env.render()
        reached = False

        for step in range(200):
            tcp_pose = to_numpy(base.agent.tcp.pose.p)  # 末端执行器位置
            error = target_pose - tcp_pose  # 计算末端执行器位置与目标位置的误差
            distance = np.linalg.norm(error)  # 计算误差的欧几里得距离

            # 每 10 步输出一次状态
            if step % 10 == 0:
                print(
            f"step={step:03d}, "
            f"distance={distance:.4f} m, "
            f"error={np.round(error, 4)}"
            )

            # 如果距离小于容差，说明已经接近目标点
            if distance < tolerance:
                print(f"已到达目标附近，误差：{distance:.4f} m")
                reached = True
                break

            # 计算动作，使用比例控制器将误差转换为末端执行器的移动指令,np.clip() 的作用就是截断超出范围的数值。
            move = np.clip(
            gain * error,
            -max_action,
            max_action,)   

            # 加上夹爪指令，整理成环境需要的形状,第四个数 1.0 表示保持夹爪张开。
            action = np.array(
            [move[0], move[1], move[2], 1.0],dtype=np.float32,).reshape(env.action_space.shape)

            # 执行动作，获取反馈
            obs, reward, terminated, truncated, info = env.step(action)

            # 渲染环境，显示机械臂动作
            env.render()
            time.sleep(0.1)  # 控制渲染速度，避免动作过快，无法观察到机械臂的运动。
            if terminated.item() or truncated.item():
                print("环境已结束本轮。")
                break

        # 如果没有到达目标点，输出最终误差
        if not reached:
            final_pos = to_numpy(base.agent.tcp.pose.p)
            final_error = np.linalg.norm(target_pos - final_pos)
            print(f"本轮未确认到达，最终误差：{final_error:.4f} m")

        input("观察完成后，在终端按 Enter 关闭窗口……")

    finally:
        env.close()


if __name__ == "__main__":
    main()

        
