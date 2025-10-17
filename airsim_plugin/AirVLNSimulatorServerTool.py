import argparse
import threading
import msgpackrpc
from pathlib import Path
import glob
import time
import os
import json
import sys
import subprocess
import errno
import signal
import copy

# 初始化一个  airsims settings 模板
AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md", # 参考文档
    "SettingsVersion": 1.2, # 版本号
    "SimMode": "ComputerVision", # ComputerVision / Multirotor
    "ViewMode": "NoDisplay", # Fpv / NoDisplay
    "ClockSpeed": 1, # 时钟速度
    # "LocalHostIp": "127.0.0.1",
    # "ApiServerPort": 10000,
    "CameraDefaults": { # 相机默认参数
        "CaptureSettings": [ # 相机捕获设置
            {
                "ImageType": 0, # 相机编号
                "Width": 224, # 图像宽度
                "Height": 224, # 图像高度
                "FOV_Degrees": 90, # 视场角
                "AutoExposureMaxBrightness": 1, # 自动曝光最大亮度
                "AutoExposureMinBrightness": 0.03 # 自动曝光最小亮度
            },
            {
                "ImageType": 2,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            },
            {
                "ImageType": 3,
                "Width": 256,
                "Height": 256,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03
            }
        ],
        "X": 0, # 相机位置X
        "Y": 0,
        "Z": 0,
        "Pitch": 0, # 相机姿态
        "Roll": 0,
        "Yaw": 0
    },
    "Recording": {
        "RecordInterval": 0.001,
        "Enabled": False, #不启用录制
        "Cameras": []
    },
    "SubWindows": [], # 子窗口设置
    "Vehicles": {} # 车辆设置
}

# 创建无人机函数
# 传入参数：每个环境的无人机数量，是否显示场景，是否为无人机模式
def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=False) -> dict:
    # deep copy 模板，防止修改原始模板
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)

    if show_scene == True:
        airsim_settings['ViewMode'] = 'Fpv'
    else:
        airsim_settings['ViewMode'] = 'NoDisplay' # 不显示场景

    if uav_mode == True:
        airsim_settings['SimMode'] = 'Multirotor'
        airsim_settings['PhysicsEngineName'] = 'ExternalPhysicsEngine'
    else:
        airsim_settings['SimMode'] = 'ComputerVision' # 这个模式只有相机，不渲染飞行器


    # create drone objects
    for i in range(drone_num_per_env):
        drone_name = 'Drone_' + str(i+1) # 每个场景里面名字都叫Drone_1

        airsim_settings['Vehicles'][str(drone_name)] = {} #占位空表

        drone = {
            "VehicleType": "ComputerVision", #相机捕获模式
            "Cameras": {
                "front_0": { # 相机名称
                    "CaptureSettings": [
                        {
                            "ImageType": 0,
                            "Width": 224,
                            "Height": 224,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        },
                        {
                            "ImageType": 2,
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        },
                        {
                            "ImageType": 3,
                            "Width": 256,
                            "Height": 256,
                            "FOV_Degrees": 90,
                            "AutoExposureMaxBrightness": 1,
                            "AutoExposureMinBrightness": 0.03
                        }
                    ],
                    "X": 0.5, "Y": 0, "Z": 0, # 相机位置
                    "Pitch": 0, "Roll": 0, "Yaw": 0
                }
            },
            "X": 0, "Y": 0, "Z": 0, # 无人机位置
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }

        #根据要求修改无人机类型
        if airsim_settings['SimMode'] == 'ComputerVision':
            drone['VehicleType'] = 'ComputerVision'
        elif airsim_settings['SimMode'] == 'Multirotor':
            drone['VehicleType'] = 'SimpleFlight' #渲染简单的飞行器
        else:
            raise NotImplementedError

        # 填入
        airsim_settings['Vehicles'][str(drone_name)] = copy.deepcopy(drone)

    return airsim_settings

# 检查本机上是否存在指定pid的进程
def pid_exists(pid) -> bool:
    """
    Check whether pid exists in the current process table.
    UNIX only.
    """
    if pid < 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        return True

# 服务端，从连接端口获取模拟器进程id
# 传入参数：端口号
# 返回值：进程id
def FromPortGetPid(port: int):
    # 设置命令
    subprocess_execute = "netstat -nlp | grep {}".format(
        port,
    )

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'FromPortGetPid',
                e,
            )
        )
        return None
    except:
        return None

    pid = None
    #一个端口号可能有多个进程占用，找到第一个tcp的即可
    for line in iter(p.stdout.readline, b''):
        line = str(line, encoding="utf-8")
        if 'tcp' in line:
            pid = line.strip().split()[-1].split('/')[0]
            try:
                pid = int(pid)
            except:
                pid = None
            break

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        # 再关掉启动子任务的这个进程本身
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass
    
    # 返回pid               
    return pid

# 本机上杀掉指定进程id的进程
# 传入参数：进程id
def KillPid(pid) -> None:
    #检查这个pid是否是整数
    if pid is None or not isinstance(pid, int):
        print('pid is not int')
        return

    while pid_exists(pid): # 如果pid有效
        try:
            # os.system(("kill -9 {}".format(pid)))
            os.kill(pid, signal.SIGKILL) # 就杀掉
        except Exception as e:
            pass
        time.sleep(0.5) # 等待0.5秒

    return

# 本机上杀掉指定端口号的进程
# 传入参数：端口号列表（很多端口）
def KillPorts(ports) -> None:
    threads = []

    # 每个子线程通过一个端口获得一个pid并杀掉
    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

    # 多线程杀掉多个端口
    for index, port in enumerate(ports):
        thread = threading.Thread(target=_kill_port, args=(index, port))
        threads.append(thread)
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    for thread in threads:
        thread.join()
    threads = []

    return

# 本机上杀掉所有AirVLN进程
def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN" # pkill用于杀掉指定名称的所有进程

    try:
        p = subprocess.Popen(
            subprocess_execute,
            stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            shell=True,
        )
    except Exception as e:
        print(
            "{}\t{}\t{}".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                'KillAirVLN',
                e,
            )
        )
        return
    except:
        return

    try:
        # os.system(("kill -9 {}".format(p.pid)))
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    time.sleep(1)
    return

# socket事件处理类
class EventHandler(object):
    # 初始化
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports # 场景端口列表，从server port开始依次加1，保留1000个端口号供场景使用

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus # 场景GPU列表，循环使用传入的GPU id列表，保留100个供场景使用

        self.scene_used_ports = [] # 已使用的场景端口列表，空表

    # 暴露出来的ping函数
    def ping(self) -> bool:
        return True

    # 打开场景函数，传入ip和场景id列表
    def _open_scenes(self, ip: str , scen_ids: list):
        print(
            "{}\tSTART closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        # 关闭所有已使用的场景端口下的进程
        KillPorts(self.scene_used_ports)
        self.scene_used_ports = []
        # KillAirVLN()
        print(
            "{}\tEND closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )


        # Occupied airsim port 1
        ports = []
        index = 0
        while len(ports) < len(scen_ids):
            # 从场景端口列表中找到未被占用的端口
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])
            index += 1

        # 杀掉这些未被占用的端口
        KillPorts(ports)


        # Occupied GPU 2
        gpus = [self.scene_gpus[index] for index in range(len(scen_ids))]


        # search scene path 3
        # 根据场景id列表找到对应的场景可执行文件路径
        choose_env_exe_paths = []
        for scen_id in scen_ids:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue

            res = glob.glob((str(SEARCH_ENVs_PATH) + '/**/' + 'env_' + str(scen_id) + '/LinuxNoEditor/AirVLN.sh'), recursive=True)
            if len(res) > 0:
                choose_env_exe_paths.append(res[0]) # 放入路径
            else:
                print(f'can not find scene file: {scen_id}')
                raise KeyError


        p_s = []
        for index in range(len(scen_ids)):
            # airsim settings 4
            airsim_settings = create_drones() # 创建无人机设置
            airsim_settings['ApiServerPort'] = int(ports[index]) # 设置API端口
            airsim_settings_write_content = json.dumps(airsim_settings) #将python字典转为json字符串
            #如果没有settings文件夹就创建一个
            if not os.path.exists(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1))):
                os.makedirs(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1)), exist_ok=True)
            with open(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'), 'w', encoding='utf-8') as dump_f:
                # 并在指定路径下写入settings.json文件
                dump_f.write(airsim_settings_write_content)


            # open scene 5
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                # 构造启动命令
                # -RenderOffscreen: 离屏渲染
                # -NoSound: 无声模式
                # -NoVSync: 关闭垂直同步
                # -GraphicsAdapter={}: 指定使用的GPU
                # --settings {} : 指定使用的settings.json文件路径
                subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} --settings {} ".format(
                    choose_env_exe_paths[index], #前半段命令
                    gpus[index],
                    str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'),
                )

                try:
                    # 启动子进程
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    # 放入进程列表
                    p_s.append(p)
                except Exception as e:
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        
        # 全部命令启动完后等待3秒
        time.sleep(3)

        # check
        threads = []

        # 检查每个场景是否成功打开（我感觉这个函数有点问题）
        def _check_scene(index, p):
            if p is None:
                print(
                    "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                        str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                        index,
                        None,
                        gpus[index],
                    )
                )
                return

            for line in iter(p.stdout.readline, b''):
                if 'Drone_' in str(line): # 如果进程的输出中包含Drone_，说明场景成功打开
                    break
            
            #不然的话，就杀掉这个进程
            try:
                p.terminate()
                # os.system(("kill -9 {}".format(p.pid)))
                os.kill(p.pid, signal.SIGKILL)
            except:
                pass
            

            print(
                "{}\tOpening {}-th scene (scene {})\tgpu:{}".format(
                    str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                    index,
                    scen_ids[index],
                    gpus[index],
                )
            )
            return

        # 多线程检查多个场景
        for index, p in enumerate(p_s):
            thread = threading.Thread(target=_check_scene, args=(index, p))
            threads.append(thread)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        threads = []

        # ChangeNice(ports)

        self.scene_used_ports += copy.deepcopy(ports)

        # 返回ip和端口号列表
        return True, (ip, ports)

    # 重新打开场景函数（暴露）
    def reopen_scenes(self, ip: str, scen_ids: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            result = self._open_scenes(ip, scen_ids) #先杀、占用、找路径、启动、检查
        except Exception as e:
            print(e)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

    # 关闭sence_used_ports对应的所有场景（暴露）
    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            KillPorts(self.scene_used_ports)
            self.scene_used_ports = []
            # KillPorts(self.scene_ports)
            # KillAirVLN()

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result

# 一个后台进程
def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t

# msgpackrpc服务端启动函数
def serve(daemon=False):
    try:
        server = msgpackrpc.Server(EventHandler())
        addr = msgpackrpc.Address(HOST, PORT)
        server.listen(addr)

        thread = serve_background(server, daemon)

        return addr, server, thread #返回ip地址、服务端对象、后台进程对象
    except Exception as err:
        print(err)
        pass


if __name__ == '__main__':
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help='server port'
    )
    args = parser.parse_args()


    HOST = '127.0.0.1' # 本机ip
    PORT = int(args.port) # 端口号，默认30000

    CWD_DIR = Path(str(os.getcwd())).resolve()  # 当前工作目录
    PROJECT_ROOT_DIR = CWD_DIR.parent           # 项目根目录 AirVLN_ws
    SEARCH_ENVs_PATH = PROJECT_ROOT_DIR / 'ENVs'# 场景文件夹路径 AirVLN_ws/ENVs
    assert os.path.exists(str(SEARCH_ENVs_PATH)), 'error' #确保场景文件夹存在

    gpu_list = []
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy() #把传入的参数转换为GPU id列表


    addr, server, thread = serve() # 启动服务端，开始监听
    print(f"start listening \t{addr._host}:{addr._port}")

