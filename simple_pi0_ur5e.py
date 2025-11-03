#!/usr/bin/env python3
import time
import torch
import numpy as np
import cv2
import pyrealsense2 as rs
import json
import subprocess
import os
import argparse
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def run_command(command_args, shell=False):
    """运行命令并返回输出"""
    try:
        result = subprocess.run(
            command_args, 
            shell=shell, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"命令执行错误: {e}")
        print(f"错误输出: {e.stderr}")
        return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用PI0模型和RealSense相机控制UR5e机器人')
    parser.add_argument('--fps', type=float, default=10, help='推理帧率')
    parser.add_argument('--duration', type=int, default=60, help='运行时间(秒)')
    parser.add_argument('--task', type=str, default='go to the yellow banana', help='任务提示词')
    parser.add_argument('--device', type=str, default='cuda', help='推理设备 (cuda, cpu, mps)')
    parser.add_argument('--sim', action='store_true', help='仿真模式 (不发送命令到机器人)')
    parser.add_argument('--scale', type=float, default=0.01, help='动作比例因子')
    parser.add_argument('--width', type=int, default=640, help='相机宽度')
    parser.add_argument('--height', type=int, default=480, help='相机高度')
    parser.add_argument('--show-image', action='store_true', help='显示相机图像')
    args = parser.parse_args()
    
    # 初始化RealSense相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, 30)
    
    print("正在启动RealSense相机...")
    pipeline.start(config)
    
    # 给相机一点时间预热
    for _ in range(5):
        pipeline.wait_for_frames()
        time.sleep(0.1)
    
    print("相机就绪")
    
    # 加载PI0策略
    print("正在加载PI0模型...")
    policy = PI0Policy.from_pretrained("lerobot/pi0")
    policy.to(args.device)
    print(f"模型已加载，配置：{policy.config}")
    
    # 开始推理循环
    total_frames = int(args.duration * args.fps)
    action_count = 0
    
    print(f"开始推理循环，将运行{args.duration}秒，共{total_frames}帧")
    print(f"任务提示词: '{args.task}'")
    print(f"{'仿真模式' if args.sim else '真实机器人模式'}")
    
    for frame_idx in range(total_frames):
        start_time = time.perf_counter()
        
        # 获取RealSense图像
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("未能获取彩色图像，跳过此帧")
            continue
        
        # 转换为numpy数组
        frame = np.asanyarray(color_frame.get_data())
        
        # BGR转RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小为224x224（模型输入尺寸）
        resized_frame = cv2.resize(frame, (224, 224))
        
        # 显示图像（如果启用）
        if args.show_image:
            cv2.imshow('RealSense Camera', cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 准备批次与观察
        batch = {}
        # 这里我们使用零张量作为关节状态，如果您有实际关节状态，可在此处替换
        batch["observation.state"] = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32).to(args.device)
        
        # 转换图像为模型所需格式
        images1 = torch.from_numpy(resized_frame).float() / 255.0
        # 我们使用相同的图像作为wrist_image，如果您有手腕相机，可在此处替换
        images2 = images1.clone()
        
        # 格式调整：添加批次维度，将通道维度放在前面
        batch["observation.images.image"] = images1.permute(2, 0, 1).unsqueeze(0).to(args.device)
        batch["observation.images.wrist_image"] = images2.permute(2, 0, 1).unsqueeze(0).to(args.device)
        batch["task"] = [args.task]
        
        # 使用策略计算下一个动作
        action = policy.select_action(batch)
        action_np = action.cpu().numpy()[0]
        
        action_count += 1
        print(f"动作 [{action_count}/{total_frames}]: {action_np}")
        
        # 如果不是仿真模式，发送命令到UR5e
        if not args.sim:
            # 将PI0输出转换为UR5e命令
            scaled_action = action_np * args.scale
            
            ur5e_command = {
                "dx": float(scaled_action[0]),
                "dy": float(scaled_action[1]),
                "dz": float(scaled_action[2]),
                "droll": float(scaled_action[3] * 10.0),  # 转换为度
                "dpitch": float(scaled_action[4] * 10.0),
                "dyaw": float(scaled_action[5] * 10.0)
            }
            
            # 构建命令行
            cmd_args = ["./relative_move_ee_6d_sim.py"]
            for key, value in ur5e_command.items():
                cmd_args.append(f"--{key[1:]} {value}")
            
            cmd_str = " ".join(cmd_args)
            print(f"执行命令: {cmd_str}")
            
            # 执行命令
            run_command(cmd_str, shell=True)
        
        # 等待下一帧
        dt_s = time.perf_counter() - start_time
        wait_time = 1 / args.fps - dt_s
        if wait_time > 0:
            busy_wait(wait_time)
        else:
            print(f"警告：处理时间({dt_s:.3f}s)超过帧间隔({1/args.fps:.3f}s)")
    
    # 清理资源
    pipeline.stop()
    if args.show_image:
        cv2.destroyAllWindows()
    
    print("程序执行完毕")


if __name__ == "__main__":
    main()