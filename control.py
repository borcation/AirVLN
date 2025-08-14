import os
import sys
import json
import time
import socket
import subprocess
import shutil
from pathlib import Path

import airsim
import numpy as np
import pygame

# ========== 可配置区域 ==========
ENV_ID = 1
AIRVLN_ROOT = Path(f"/home/vergil/AirVLN_ws/ENVs/env_{ENV_ID}/env_{ENV_ID}/LinuxNoEditor")
AIRVLN_SH = AIRVLN_ROOT / "AirVLN.sh"
SETTINGS_PATH = AIRVLN_ROOT / "settings_control.json"
API_PORT = 41451
CAM_W, CAM_H = 256, 256

# 新增：选择渲染后端与独显索引
BACKEND = "opengl"   # 备选: "opengl"
GPU_INDEX = 1        # 尝试 0/1，取决于你机器上的设备顺序
# =================================


def write_settings(path: Path, api_port: int):
    settings = {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",         # 使用无人机模式，便于速度控制
        "ViewMode": "Fpv",               # 可见窗口
        "ClockSpeed": 1,
        "LocalHostIp": "127.0.0.1",
        "ApiServerPort": api_port,
        "CameraDefaults": {
            "CaptureSettings": [
                {"ImageType": 0, "Width": CAM_W, "Height": CAM_H, "FOV_Degrees": 90}
            ]
        },
        "Vehicles": {
            "Drone_1": {
                "VehicleType": "SimpleFlight",
                "Cameras": {
                    "front_0": {
                        "CaptureSettings": [
                            {"ImageType": 0, "Width": CAM_W, "Height": CAM_H, "FOV_Degrees": 90}
                        ]
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def wait_port(host: str, port: int, timeout: int = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(1)
    return False


def launch_env(airvln_sh: Path, settings_path: Path) -> subprocess.Popen:
    assert airvln_sh.exists(), f"找不到启动脚本: {airvln_sh}"

    cmd_prefix = []
    if shutil.which("prime-run"):
        cmd_prefix = ["prime-run"]

    env = os.environ.copy()
    env.setdefault("SDL_VIDEODRIVER", "x11")
    env.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    env.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    env.setdefault("__VK_LAYER_NV_optimus", "NVIDIA_only")

    backend_flag = ["-vulkan"] if BACKEND.lower() == "vulkan" else ["-opengl4"]
    graphics_flag = [f"-GraphicsAdapter={GPU_INDEX}"]

    cmd = [
        *cmd_prefix,
        "bash", str(airvln_sh),
        "-NoSound",
        "-NoVSync",
        "--settings", str(settings_path.resolve()),
        *backend_flag,
        *graphics_flag,
    ]
    print("启动命令:", " ".join(cmd))
    # 不读取输出，避免阻塞
    p = subprocess.Popen(cmd, env=env)
    return p


def connect_airsim(port: int) -> airsim.MultirotorClient:
    # 等待端口真正监听，避免 ECONNREFUSED
    if not wait_port("127.0.0.1", port, timeout=90):
        raise RuntimeError(f"AirSim 端口 {port} 长时间未开放（引擎可能未启动或已崩溃）")

    client = airsim.MultirotorClient(ip="127.0.0.1", port=port)
    # 进一步用一个 RPC 做实连通验证
    for _ in range(10):
        try:
            _ = client.getServerVersion()  # 或 client.simGetWorldExtents()
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("AirSim RPC 未响应（服务可能刚起或已退出）")

    client.enableApiControl(True)
    client.armDisarm(True)
    try:
        client.takeoffAsync(timeout_sec=6).join()
    except Exception:
        pass
    return client


def get_rgb(client: airsim.MultirotorClient) -> np.ndarray:
    req = airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
    resp = client.simGetImages([req])[0]
    if resp.height == 0 or resp.width == 0 or len(resp.image_data_uint8) == 0:
        raise RuntimeError(f"空图像: h={resp.height}, w={resp.width}, bytes={len(resp.image_data_uint8)}")
    img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(resp.height, resp.width, 3)
    return img


def send_velocity(client: airsim.MultirotorClient, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0, dur=0.1):
    """
    优先使用机体坐标系速度（若可用），否则退化到世界坐标系。
    """
    try:
        # 新版 API（机体坐标系），更符合直觉：x前后、y左右、z上下（z向下为正）
        client.moveByVelocityBodyFrameAsync(vx, vy, vz, dur, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))
    except AttributeError:
        # 旧版 AirSim 没有 BodyFrame API，退化到世界坐标系（简单用当前朝向近似）
        client.moveByVelocityAsync(vx, vy, vz, dur, yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))


def main():
    # 写入 AirSim 设置
    write_settings(SETTINGS_PATH, API_PORT)

    # 启动环境
    proc = launch_env(AIRVLN_SH, SETTINGS_PATH)

    # 等待环境加载
    print("等待环境初始化...")
    time.sleep(8)

    # 连接 AirSim
    client = connect_airsim(API_PORT)
    print("已连接 AirSim，键位：W/S 前后，A/D 左右，Q/E 上下，J/L 左右旋转，ESC 退出。")

    # 初始化 pygame 可视化窗口
    pygame.init()
    win = pygame.display.set_mode((CAM_W, CAM_H), pygame.RESIZABLE)
    pygame.display.set_caption("AirVLN 可视化与键盘控制")

    clock = pygame.time.Clock()
    win_size = [CAM_W, CAM_H]

    try:
        while True:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if event.type == pygame.VIDEORESIZE:
                    win_size = [event.w, event.h]
                    win = pygame.display.set_mode(win_size, pygame.RESIZABLE)

            # 按键状态（支持长按）
            keys = pygame.key.get_pressed()
            vx = vy = vz = 0.0
            yaw_rate = 0.0

            speed = 1.5     # m/s
            vz_speed = 1.0  # m/s
            yaw_deg = 20.0  # deg/s

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                vx += speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                vx -= speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                vy -= speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                vy += speed
            if keys[pygame.K_q]:
                vz -= vz_speed   # AirSim: z 向下为正，这里用负号表示上升
            if keys[pygame.K_e]:
                vz += vz_speed   # 下降
            if keys[pygame.K_j]:
                yaw_rate -= yaw_deg
            if keys[pygame.K_l]:
                yaw_rate += yaw_deg

            # 发送控制（小步长，平滑连续）
            send_velocity(client, vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate, dur=0.1)

            # 拉取并显示相机
            try:
                img = get_rgb(client)
                surf = pygame.surfarray.make_surface(np.rot90(img))
                surf = pygame.transform.smoothscale(surf, win_size)
                win.blit(surf, (0, 0))
                pygame.display.update()
            except Exception as e:
                pass

            clock.tick(30)  # 30 FPS 循环

    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        pygame.quit()
        print("已退出。")


if __name__ == "__main__":
    main()