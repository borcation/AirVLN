import msgpackrpc
import time
import airsim
import threading
import random
import copy
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import sys
    cur_path=os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, cur_path+"/..")
    print(os.getcwd())
    from src.common.param import args
else:
    from src.common.param import args

from utils.logger import logger

# 多线程启动
class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.flag_ok = False

    def run(self):
        try:
            self.result = self.func(*self.args)
        except Exception as e:
            logger.error(e)
            self.flag_ok = False
        else:
            self.flag_ok = True

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except:
            return None

# 客户端工具类
'''
- socket 连接的作用：作为控制通道，远程机器上有一个 msgpackrpc 服务（控制程序）。客户端通过 socket 调用 reopen_scenes / close_scenes 来让远程启动/重启对应的 Unreal/AirSim 场景，并返回每个场景对应的 IP/port。也就是“下发打开场景的命令并获取实际监听端口”。
- AirSim（VehicleClient）连接的作用：是真正用来获取图像、设置 Pose 的 RPC 客户端，直接连接到每个场景对应的 AirSim RPC 端口（ip:port）。

必须先用 socket 命令让远端启动（或重启）场景并告诉你每个场景的端口，才能用 airsim.VehicleClient 连接到正确的端口。
建立好所有 VehicleClient 后，socket 控制通道就暂时不需要了，所以关闭以释放资源；后续需要控制远端再打开/关闭场景时会重新新建 socket。

run_call 里先 _closeConnection() 是为了清理已有的 AirSim 客户端，保证“干净”重启场景。
多线程并发调用 reopen_scenes 是为了并行启动多台机器上的多个场景并减少等待时间。

'''
class AirVLNSimulatorClientTool:
    # 初始化：机器信息、socket信息、airsim客户端信息、连接检查
    def __init__(self, machines_info) -> None:
        self.machines_info = copy.deepcopy(machines_info)
        # 一个表是socket连接，一个表是airsim连接
        self.socket_clients = []
        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in machines_info ]

        self._init_check()

    # 内部方法，初始化检查：machines_info中的IP地址不重复
    def _init_check(self) -> None:
        ips = [item['MACHINE_IP'] for item in self.machines_info]
        assert len(ips) == len(set(ips)), 'MACHINE_IP repeat'

    # 内部方法，确认socket连接，用ping方法，返回true或false
    def _confirmSocketConnection(self, socket_client: msgpackrpc.Client) -> bool:
        try:
            socket_client.call('ping')
            logger.info("Connected\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            return True
        except:
            try:
                logger.error("Ping returned false\t{}:{}".format(socket_client.address._host, socket_client.address._port))
            except:
                logger.error('Ping returned false')
            return False
    
    # 内部方法，确认airsim连接
    def _confirmConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    self.airsim_clients[index_1][index_2].confirmConnection()

        return

    # 内部方法，关闭socket连接
    def _closeSocketConnection(self) -> None:
        socket_clients = self.socket_clients

        for socket_client in socket_clients:
            try:
                socket_client.close()
            except Exception as e:
                pass

        self.socket_clients = []
        return
    # 内部方法，关闭airsim连接
    def _closeConnection(self) -> None:
        for index_1, _ in enumerate(self.airsim_clients):
            for index_2, _ in enumerate(self.airsim_clients[index_1]):
                if self.airsim_clients[index_1][index_2] is not None:
                    try:
                        self.airsim_clients[index_1][index_2].close()
                    except Exception as e:
                        pass

        self.airsim_clients = [[None for _ in list(item['open_scenes'])] for item in self.machines_info]
        return

    # 运行调用，建立socket连接，关闭airsim连接，开启场景，确认连接，关闭socket连接
    def run_call(self, airsim_timeout: int=60) -> None:
        # 通过msgpackrpc建立socket连接
        socket_clients = []
        for index, item in enumerate(self.machines_info):
            socket_clients.append(
                msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
            )
        # 确认socket连接正常
        for socket_client in socket_clients:
            if not self._confirmSocketConnection(socket_client):
                logger.error('cannot establish socket')
                raise Exception('cannot establish socket')
        # 保存socket连接为成员变量
        self.socket_clients = socket_clients

        # 关闭所有airsim连接（清空表）
        before = time.time()
        self._closeConnection()


        # 通过socket连接，开启场景
        def _run_command(index, socket_client: msgpackrpc.Client):
            # 开起场景失败，就重试
            logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            result = socket_client.call('reopen_scenes', socket_client.address._host, self.machines_info[index]['open_scenes'])

            # 重试失败，报错
            if result[0] == False:
                logger.error(f'Failed to open scenes, machine : {socket_client.address._host}:{socket_client.address._port}')
                raise Exception('Failed to open scenes')
            assert len(result[1]) == 2, 'Failed to open scenes'

            # 开起场景成功，保存ip和端口，建立airsim连接
            ip = result[1][0]
            ports = result[1][1]

            if isinstance(ip, bytes):
                ip = ip.decode()

            # 检查ip地址不变，端口数量和open_scenes数量相同
            assert str(ip) == str(socket_client.address._host), 'Failed to open scenes'
            assert len(ports) == len(self.machines_info[index]['open_scenes']), 'Failed to open scenes'

            # 依次登记，建立airsim连接
            for i, port in enumerate(ports):
                if self.machines_info[index]['open_scenes'][i] is None:
                    self.airsim_clients[index][i] = None
                else:
                    self.airsim_clients[index][i] = airsim.VehicleClient(ip=ip, port=port, timeout_value=airsim_timeout)

            #下面这个logger是不是有问题？还是failed是对的？
            #logger.info(f'Failed to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            logger.info(f'Success to open scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
            return

        # 建立线程池临时变量，准备开启场景
        threads = []
        thread_results = []
        for index, socket_client in enumerate(socket_clients):
            threads.append(
                # 每个子现场程运行_run_command函数，传入index和socket_client，负责一个场景的开启
                MyThread(_run_command, (index, socket_client))
            )
        # 启动线程池
        for thread in threads:
            thread.setDaemon(True)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            thread.get_result()
            thread_results.append(thread.flag_ok)
        # 清空线程池变量
        threads = []
        # 检查所有场景是否开启成功
        if not (np.array(thread_results) == True).all():
            raise Exception('Failed to open scenes')

        # 记录时间
        after = time.time()
        diff = after - before
        logger.info(f"Start time: {diff}")

        # 确认airsim连接
        self._confirmConnection()
        # 关闭socket连接
        self._closeSocketConnection()

    # 获取图像响应，返回图像
    def getImageResponses(self, get_rgb=True, get_depth=True):

        # 通过airsim连接获取图像
        def _getImages(airsim_client: airsim.VehicleClient, scen_id, get_rgb, get_depth):
            if airsim_client is None:
                raise Exception('error')
                return None, None

            img_rgb = None
            img_depth = None

            if not get_rgb and not get_depth:
                return None, None

            if scen_id in [1, 7]:
                time_sleep_cnt = 0
                while True:
                    try:
                        # 构造ImageRequest列表，请求"front_0"相机的图像
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthVis, pixels_as_float=False, compress=True)
                            )

                        # 发送请求，获取图像响应
                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break

                        # 处理收到的图像
                        img_rgb = None
                        img_depth = None

                        if get_rgb:
                            # 检查图像尺寸是否正确
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            # 一维数组化图像数据
                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            if args.run_type not in ['eval']:
                                # 如果不是eval模式，检查图像内容是否有效：取平坦化后第一个像素值，判断是否所有像素值都相同，如果都相同，说明图像错误，则报错
                                assert not (img1d.flatten()[0] == img1d).all(), 'Failed to retrieve RGB image'
                            # 重塑为三维数组（高度，宽度，通道数）
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            # 转换为numpy格式，数组形状不变
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'
                            
                            # 将深度图像数据保存为临时PNG文件，然后读取该文件
                            png_file_name = '/tmp/AirVLN_depth_{}_{}.png'.format(time.time(), random.randint(0, 10000))
                            airsim.write_file(png_file_name, response_depth.image_data_uint8)
                            # 使用cv2接口读取PNG文件
                            img3d = cv2.imread(png_file_name)
                            # 删除临时PNG文件
                            os.remove(png_file_name)
                            # 提取第二个通道的数据作为深度图像
                            img1d = img3d[:, :, 1]
                            # 重塑为三维数组（高度，宽度，1）
                            img1d = img1d.reshape(response_depth.height, response_depth.width, 1)
                            # 将像素值归一化到0-1之间，每个值都除以255
                            obs_depth_img = img1d / 255
                            
                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        # 如果assert报错，说明图像获取失败，休眠1秒后重试
                        time_sleep_cnt += 1
                        logger.error("Image retrieval error")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    # 超过20次重试仍然失败（20s），抛出异常
                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            else:
                # 场景ID不是1或7时，使用透视深度图
                time_sleep_cnt = 0
                while True:
                    try:
                        ImageRequest = []
                        if get_rgb:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
                            )
                        if get_depth:
                            ImageRequest.append(
                                airsim.ImageRequest("front_0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)
                            )

                        responses = airsim_client.simGetImages(ImageRequest, vehicle_name='Drone_1')

                        if get_rgb and get_depth:
                            response_rgb = responses[0]
                            response_depth = responses[1]
                        elif get_rgb and not get_depth:
                            response_rgb = responses[0]
                        elif not get_rgb and get_depth:
                            response_depth = responses[0]
                        else:
                            break

                        if get_rgb:
                            assert response_rgb.height == args.Image_Height_RGB and response_rgb.width == args.Image_Width_RGB, 'Failed to retrieve RGB image'

                            img1d = np.frombuffer(response_rgb.image_data_uint8, dtype=np.uint8)
                            img_rgb = img1d.reshape(response_rgb.height, response_rgb.width, 3)
                            img_rgb = np.array(img_rgb)

                        if get_depth:
                            assert response_depth.height == args.Image_Height_DEPTH and response_depth.width == args.Image_Width_DEPTH, 'Failed to retrieve DEPTH image'

                            # 将深度图像数据转换为二维浮点数组，单位为米
                            depth_img_in_meters = airsim.list_to_2d_float_array(response_depth.image_data_float, response_depth.width, response_depth.height)
                            # 检查深度图像内容是否有效
                            if depth_img_in_meters.min() < 1e4:
                                assert not (depth_img_in_meters.flatten()[0] == depth_img_in_meters).all(), 'Failed to retrieve DEPTH image'
                            # 重塑为三维数组（高度，宽度，1）
                            depth_img_in_meters = depth_img_in_meters.reshape(response_depth.height, response_depth.width, 1)
                            # 将深度值裁剪到0-100米之间，并归一化到0-1之间 
                            obs_depth_img = np.clip(depth_img_in_meters, 0, 100)
                            obs_depth_img = obs_depth_img / 100

                            img_depth = np.array(obs_depth_img, dtype=np.float32)

                        break
                    except:
                        time_sleep_cnt += 1
                        logger.error("Failed to retrieve image")
                        logger.error('time_sleep_cnt: {}'.format(time_sleep_cnt))
                        time.sleep(1)

                    if time_sleep_cnt > 20:
                        raise Exception('Failed to retrieve image')

            # Tip: If you are using AirVLN code for the first time, please confirm that the
            #       channel order of the images captured is as expected by visualization!
            # Example is as below:

            # plt.imsave('./tmp/img_rgb.png', img_rgb)

            # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            # plt.imsave('./tmp/img_rgb_converted.png', img_rgb)

            # plt.imsave('./tmp/img_depth.png', img_depth.squeeze(), cmap='gray')

            return img_rgb, img_depth

        # 获取图像的多线程调用
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    # 每个子现场程运行_getImages函数，传入airsim_client和scene_id，负责一个场景的图像获取
                    MyThread(_getImages, (self.airsim_clients[index_1][index_2], self.machines_info[index_1]['open_scenes'][index_2], get_rgb, get_depth))
                )
        # 启动线程池
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        # 收集结果
        responses = []
        for index_1, _ in enumerate(threads):
            responses.append([])
            for index_2, _ in enumerate(threads[index_1]):
                responses[index_1].append(
                    threads[index_1][index_2].get_result()
                )
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('getImageResponses failed')
            return None

        return responses

    # 设置位姿，返回bool
    def setPoses(self, poses: list) -> bool:

        # 通过airsim连接设置位姿
        def _setPoses(airsim_client: airsim.VehicleClient, pose: airsim.Pose) -> None:
            if airsim_client is None:
                raise Exception('error')
                return

            # 调用simSetVehiclePose接口设置位姿
            airsim_client.simSetVehiclePose(
                pose=pose,
                ignore_collision=True,
                vehicle_name='Drone_1',
            )

            return

        # 多线程应用
        threads = []
        thread_results = []
        for index_1 in range(len(self.airsim_clients)):
            threads.append([])
            for index_2 in range(len(self.airsim_clients[index_1])):
                threads[index_1].append(
                    MyThread(_setPoses, (self.airsim_clients[index_1][index_2], poses[index_1][index_2]))
                )
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].setDaemon(True)
                threads[index_1][index_2].start()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].join()
        for index_1, _ in enumerate(threads):
            for index_2, _ in enumerate(threads[index_1]):
                threads[index_1][index_2].get_result()
                thread_results.append(threads[index_1][index_2].flag_ok)
        threads = []
        if not (np.array(thread_results) == True).all():
            logger.error('setPoses failed')
            return False

        return True

    # 关闭场景
    def closeScenes(self):
        try:
            # 通过msgpackrpc建立socket连接
            socket_clients = []
            for index, item in enumerate(self.machines_info):
                socket_clients.append(
                    msgpackrpc.Client(msgpackrpc.Address(item['MACHINE_IP'], item['SOCKET_PORT']), timeout=180)
                )

            # 确认socket连接正常
            for socket_client in socket_clients:
                if not self._confirmSocketConnection(socket_client):
                    logger.error('cannot establish socket')
                    raise Exception('cannot establish socket')

            self.socket_clients = socket_clients

            # 先关闭所有airsim连接（清空表）
            self._closeConnection()

            # 通过socket连接，关闭场景
            def _run_command(index, socket_client: msgpackrpc.Client):
                logger.info(f'START closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                result = socket_client.call('close_scenes', socket_client.address._host)
                logger.info(f'END closing all scenes, machine {index}: {socket_client.address._host}:{socket_client.address._port}')
                return

            # 多线程关闭场景
            threads = []
            for index, socket_client in enumerate(socket_clients):
                threads.append(
                    MyThread(_run_command, (index, socket_client))
                )
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
            threads = []

            # 关闭socket连接
            self._closeSocketConnection()
        except Exception as e:
            logger.error(e)

# 测试代码
if __name__ == '__main__':

    machines_info_xxx = [
        {
            'MACHINE_IP': '127.0.0.1',
            'SOCKET_PORT': 30000,
            'MAX_SCENE_NUM': 1,
            'open_scenes': [1],
        },
    ]
    # machines_info_xxx = [
    #     {
    #         'MACHINE_IP': '127.0.0.1',
    #         'SOCKET_PORT': 30000,
    #         'MAX_SCENE_NUM': 8,
    #         'open_scenes': [1, 2, 3, 4, 5, 6, 7, None],
    #     },
    # ]

    tool = AirVLNSimulatorClientTool(machines_info=machines_info_xxx)
    tool.run_call()

    start_time = time.time()
    while True:
        time_1 = time.time()
        # 测试获取图像
        responses = tool.getImageResponses()
        time_2 = time.time()
        print(
            "total_time: {} \t time: {} \t fps: {}".format(
                (time_2-start_time),
                (time_2-time_1),
                1/(time_2-time_1),
            )
        )

        poses = []
        # poses是个二维数组，表示第index_1台机器的第index_2个场景的pose
        for index_1, item in enumerate(machines_info_xxx):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                pose=airsim.Pose(
                    # 随机生成位置
                    position_val=airsim.Vector3r(random.randint(0, 1000), random.randint(0, 1000), random.randint(-200, 0)),
                    # 固定朝向（四元数）
                    orientation_val=airsim.Quaternionr(0, 0, 0, 1),
                )
                poses[index_1].append(pose)
        # 测试设置位姿
        tool.setPoses(poses)

