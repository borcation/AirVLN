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


AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "ComputerVision", # ComputerVision / Multirotor
    "ViewMode": "NoDisplay", # Fpv / NoDisplay
    "ClockSpeed": 1,
    # "LocalHostIp": "127.0.0.1",
    # "ApiServerPort": 10000,
    "CameraDefaults": {
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
        "X": 0,
        "Y": 0,
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0
    },
    "Recording": {
        "RecordInterval": 0.001,
        "Enabled": False,
        "Cameras": []
    },
    "SubWindows": [],
    "Vehicles": {}
}


def create_drones(drone_num_per_env=1, show_scene=False, uav_mode=False) -> dict:
    airsim_settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)

    if show_scene == True:
        airsim_settings['ViewMode'] = 'Fpv'
    else:
        airsim_settings['ViewMode'] = 'NoDisplay'

    if uav_mode == True:
        airsim_settings['SimMode'] = 'Multirotor'
        airsim_settings['PhysicsEngineName'] = 'ExternalPhysicsEngine'
    else:
        airsim_settings['SimMode'] = 'ComputerVision'


    # create drone objects
    for i in range(drone_num_per_env):
        drone_name = 'Drone_' + str(i+1)

        airsim_settings['Vehicles'][str(drone_name)] = {}

        drone = {
            "VehicleType": "ComputerVision",
            "Cameras": {
                "front_0": {
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
                    "X": 0.5, "Y": 0, "Z": 0,
                    "Pitch": 0, "Roll": 0, "Yaw": 0
                }
            },
            "X": 0, "Y": 0, "Z": 0,
            "Pitch": 0, "Roll": 0, "Yaw": 0
        }

        if airsim_settings['SimMode'] == 'ComputerVision':
            drone['VehicleType'] = 'ComputerVision'
        elif airsim_settings['SimMode'] == 'Multirotor':
            drone['VehicleType'] = 'SimpleFlight'
        else:
            raise NotImplementedError

        airsim_settings['Vehicles'][str(drone_name)] = copy.deepcopy(drone)

    return airsim_settings


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


def FromPortGetPid(port: int):
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
        os.kill(p.pid, signal.SIGKILL)
    except:
        pass

    return pid


def KillPid(pid) -> None:
    if pid is None or not isinstance(pid, int):
        print('pid is not int')
        return

    while pid_exists(pid):
        try:
            # os.system(("kill -9 {}".format(pid)))
            os.kill(pid, signal.SIGKILL)
        except Exception as e:
            pass
        time.sleep(0.5)

    return


def KillPorts(ports) -> None:
    threads = []

    def _kill_port(index, port):
        pid = FromPortGetPid(port)
        KillPid(pid)

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


def KillAirVLN() -> None:
    subprocess_execute = "pkill -9 AirVLN"

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


class EventHandler(object):
    def __init__(self):
        scene_ports = []
        for i in range(1000):
            scene_ports.append(
                int(args.port) + (i+1)
            )
        self.scene_ports = scene_ports  # 预生成1000个可用端口号（基于启动参数port）

        scene_gpus = []
        while len(scene_gpus) < 100:
            scene_gpus += GPU_IDS.copy()
        self.scene_gpus = scene_gpus  # 生成100个GPU编号（循环填充）

        self.scene_used_ports = []  # 记录当前已分配/使用的端口

    def ping(self) -> bool:
        return True  # 健康检查，客户端可用来测试服务端是否在线

    def _open_scenes(self, ip: str , scen_ids: list):
        # 打印开始关闭场景的时间和提示
        print(
            "{}\tSTART closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        # 杀死之前用过的端口对应的进程
        KillPorts(self.scene_used_ports)
        # 清空已用端口列表
        self.scene_used_ports = []
        # KillAirVLN()  # 注释掉的代码，原本用于杀死所有AirVLN进程
        # 打印结束关闭场景的时间和提示
        print(
            "{}\tEND closing scenes ".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        # 1. 分配可用的airsim端口
        ports = []
        index = 0
        # 遍历scene_ports，找到未被占用的端口，数量与scen_ids一致
        while len(ports) < len(scen_ids):
            pid = FromPortGetPid(self.scene_ports[index])
            if pid is None or not isinstance(pid, int):
                ports.append(self.scene_ports[index])  # 端口未被占用则加入
            index += 1

        # 再次确保这些端口没有被占用，杀死相关进程
        KillPorts(ports)

        # 2. 分配GPU
        # 按照场景数量分配GPU编号
        gpus = [self.scene_gpus[index] for index in range(len(scen_ids))]


        # 3. 查找每个场景的可执行文件路径
        choose_env_exe_paths = []
        for scen_id in scen_ids:
            if str(scen_id).lower() == 'none':
                choose_env_exe_paths.append(None)
                continue

            # 递归查找ENVs目录下对应场景的AirVLN.sh脚本
            res = glob.glob((str(SEARCH_ENVs_PATH) + '/**/' + 'env_' + str(scen_id) + '/LinuxNoEditor/AirVLN.sh'), recursive=True)
            if len(res) > 0:
                choose_env_exe_paths.append(res[0])  # 找到脚本则加入
            else:
                print(f'can not find scene file: {scen_id}')
                raise KeyError  # 没找到则报错


        # 4. 启动每个场景
        p_s = []
        for index in range(len(scen_ids)):
            # 生成airsim settings配置
            airsim_settings = create_drones()
            airsim_settings['ApiServerPort'] = int(ports[index])  # 设置端口
            airsim_settings_write_content = json.dumps(airsim_settings)
            # 创建settings目录
            if not os.path.exists(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1))):
                os.makedirs(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1)), exist_ok=True)
            # 写入settings.json
            with open(str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'), 'w', encoding='utf-8') as dump_f:
                dump_f.write(airsim_settings_write_content)


            # 启动场景
            if choose_env_exe_paths[index] is None:
                p_s.append(None)
                continue
            else:
                # 构造启动命令，指定GPU和settings
                subprocess_execute = "bash {} -RenderOffscreen -NoSound -NoVSync -GraphicsAdapter={} --settings {} ".format(
                    choose_env_exe_paths[index],
                    gpus[index],
                    str(CWD_DIR / 'airsim_plugin/settings' / str(index+1) / 'settings.json'),
                )

                try:
                    # 启动子进程运行场景
                    p = subprocess.Popen(
                        subprocess_execute,
                        stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        shell=True,
                    )
                    p_s.append(p)
                except Exception as e:
                    # 启动失败，打印错误并返回
                    print(
                        "{}\t{}".format(
                            str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                            e,
                        )
                    )
                    return False, None
                except:
                    return False, None
        # 等待3秒，给场景启动留时间
        time.sleep(3)

        # 5. 检查场景是否启动成功（通过线程并发检查）
        threads = []

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
                if 'Drone_' in str(line):
                    break  # 检查进程输出，出现"Drone_"说明场景已启动

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

        for index, p in enumerate(p_s):
            thread = threading.Thread(target=_check_scene, args=(index, p))
            threads.append(thread)
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        threads = []

        # ChangeNice(ports)  # 可选，调整进程优先级

        self.scene_used_ports += copy.deepcopy(ports)  # 记录已用端口

        return True, (ip, ports)  # 返回成功标志、IP和端口列表

    def reopen_scenes(self, ip: str, scen_ids: list):
        print(
            "{}\tSTART reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        try:
            result = self._open_scenes(ip, scen_ids)  # 调用_open_scenes实际打开场景
        except Exception as e:
            print(e)
            result = False, None
        print(
            "{}\tEND reopen_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result  # 返回_open_scenes的结果

    def close_scenes(self, ip: str) -> bool:
        print(
            "{}\tSTART close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )

        try:
            KillPorts(self.scene_used_ports)  # 杀死所有已用端口的进程
            self.scene_used_ports = []        # 清空已用端口记录
            # KillPorts(self.scene_ports)     # 可选，杀死所有scene_ports
            # KillAirVLN()                   # 可选，杀死所有AirVLN进程

            result = True
        except Exception as e:
            print(e)
            result = False

        print(
            "{}\tEND close_scenes".format(
                str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
            )
        )
        return result  # 返回是否成功


def serve_background(server, daemon=False):
    def _start_server(server):
        server.start()
        server.close()

    t = threading.Thread(target=_start_server, args=(server,))
    t.setDaemon(daemon)
    t.start()
    return t


def serve(daemon=False):  # 启动RPC服务的函数
    try:
        server = msgpackrpc.Server(EventHandler())  # 创建msgpackrpc的Server对象，并注册事件处理器（EventHandler类）
        addr = msgpackrpc.Address(HOST, PORT)  # 构造监听的地址和端口
        server.listen(addr)  # 让Server开始监听指定地址

        thread = serve_background(server, daemon)  # 在后台线程启动Server

        return addr, server, thread  # 返回监听地址、Server对象和线程对象
    except Exception as err:
        print(err)  # 如果出错，打印异常信息
        pass

# 其中 msgpackrpc.Server(EventHandler()) 是 msgpackrpc 库的类构造方法，不是你自己实现的，是第三方库函数。


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
    #命令解释器，用来解析命令
    #gpu为collect.sh给予的，--port是default

    HOST = '127.0.0.1'
    PORT = int(args.port)

    CWD_DIR = Path(str(os.getcwd())).resolve() #AirVLN
    PROJECT_ROOT_DIR = CWD_DIR.parent #AIRVLN_ws
    SEARCH_ENVs_PATH = PROJECT_ROOT_DIR / 'ENVs'
    assert os.path.exists(str(SEARCH_ENVs_PATH)), 'error'

    gpu_list = [] #可以多个gpu
    gpus = str(args.gpus).split(',')
    for gpu in gpus:
        gpu_list.append(int(gpu.strip()))
    GPU_IDS = gpu_list.copy()
    #GPUIDS 存入最终版的gpu目录

    addr, server, thread = serve()
    print(f"start listening \t{addr._host}:{addr._port}")#证明成功了？

