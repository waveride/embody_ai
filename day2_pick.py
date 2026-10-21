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
    return tensor[0].detach().cpu().numpy().copy()

def main():
    stage = "approach"
    closed_step = 0
    hold_step = 0
    # 创建初始环境与任务
    env = gym.make(
                "PickCube-v1",           
                robot_uids = "panda",       
                num_envs = 1,         
                sim_backend = "cpu",         
                obs_mode = "state",    
                control_mode = "pd_ee_delta_pos",   
                render_mode = "human", 
                max_episode_steps=200
    )

    try:
            obs, info = env.reset(seed=Seed)   
            base = env.unwrapped    

            # 判断底座是否旋转，如果旋转了需要对齐坐标系
            root_q = to_numpy(base.agent.robot.pose.q)  
            if not np.allclose(np.abs(root_q), [1.0, 0.0, 0.0, 0.0], atol=1e-4    ):
                raise RuntimeError("底座存在旋转，需要先转换控制误差的坐标系。")
    
            tcp_pose = to_numpy(base.agent.tcp.pose.p)  
            cube_pose = to_numpy(base.cube.pose.p)  
            target_pose = cube_pose + np.array([0.0, 0.0, 0.12], dtype=np.float32)

    
            print("观测形状：", obs.shape)
            print("动作空间：", env.action_space)
            print("初始 TCP 位置：", tcp_pose)
            print("方块中心位置：", cube_pose)
            print("今天的接近目标：", target_pose)
    
            env.render()
            reached = False

            for step in range(200):
                tcp_pose = to_numpy(base.agent.tcp.pose.p)  
                cube_pose = to_numpy(base.cube.pose.p)

                #状态定义
                if stage == "approach":
                    target_pose = cube_pose + np.array([0.0, 0.0, 0.12], dtype=np.float32)
                    gripper = 1.0

                elif stage == "descend":
                    target_pose = cube_pose + np.array([0.0, 0.0, 0.01], dtype=np.float32)
                    gripper = 1.0

                elif stage == "grasp":
                    target_pose = cube_pose + np.array([0.0, 0.0, 0.01], dtype=np.float32)
                    gripper = -1.0

                elif stage == "lift":
                    target_pose = cube_pose + np.array([0.0, 0.0, 0.12], dtype=np.float32)
                    gripper = -1.0

                elif stage == "hold":
                    target_pose = cube_pose + np.array([0.0, 0.0, 0.12], dtype=np.float32)
                    gripper = -1.0

                erro = target_pose - tcp_pose
                distance = np.linalg.norm(erro)

                # 状态判断与切换
                if stage == "approach" and distance < tolerance:
                    print("接近完成，开始下降")
                    stage = "descend"
                    continue

                if stage == "descend" and distance < tolerance:
                    print("下降完成，开始闭合夹爪")
                    stage = "grasp"
                    continue

                # 接近动作
                move = np.clip(gain * erro,-max_action,max_action,)
                action = np.array([move[0], move[1], move[2], gripper],dtype=np.float32,).reshape(env.action_space.shape)

                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.05)

                if stage == "grasp":
                    print("夹爪闭合完成，开始抬起")
                    closed_step +=1

                    if closed_step >= 10:
                        lift_target = to_numpy(base.agent.tcp.pose.p)
                        lift_target[2] += 0.12
                        lift_target[0] += 0.1
                        print("抬起目标位置：", lift_target)
                        stage = "lift"


                elif stage == "lift":
                    current_pos = to_numpy(base.agent.tcp.pose.p)
                    lift_error = lift_target - current_pos
                    if np.linalg.norm(lift_error) < tolerance:
                        print("抬起完成，开始保持")
                        stage = "hold"

                elif stage == "hold":
                    hold_step += 1
                    current_cube = to_numpy(base.cube.pose.p)
                    cube_hincrease = current_cube[2] - cube_pose[2]

                    if cube_hincrease > 0.05:
                        print("方块抬起成功！")
                        reached = True
                        break

                    if hold_step >= 20:
                        print(
                            f"通过抓取练习检查：方块持续升高超过 5 cm，"
                            f"当前升高 {height_increase:.3f} m")
                        break

                if terminated.item() or truncated.item():
                    print(f"环境结束，当前阶段：{stage}")
                    break
            else:
                print(f"达到循环上限，当前阶段：{stage}")

            input("观察完成后，在终端按 Enter 关闭窗口……")   
                        

    finally:
        env.close()


if __name__ == "__main__":
    main()
                
                


                
