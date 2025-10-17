import copy
import random
import time

import msgpack_numpy
import numpy as np
import math
from gym import spaces
import lmdb
import os
import json
from pathlib import Path
import airsim
import threading
from fastdtw import fastdtw
import tqdm

from typing import Dict, List, Optional

from src.common.param import args
from utils.logger import logger
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from utils.env_utils import SimState, getPoseAfterMakeAction
from utils.env_vector import VectorEnvUtil
from utils.shorest_path_sensor import EuclideanDistance3


def load_my_datasets(splits):
	"""
	加载并合并给定分割（splits）对应的数据集文件，支持按数量截取。
	参数:
	 - splits: 文件名 "train" 或 "train@N" 表示只取前 N 条。
	返回:
	 - data: 合并后的 episodes 列表
	 - vocab: 词表（当前实现可能为空）
	"""
	data = []
	vocab = {}
	old_state = random.getstate()
	for split in splits:
		components = split.split("@")
		number = -1
		if len(components) > 1:
			split, number = components[0], int(components[1])

		# Load Json
		with open(str(Path(args.project_prefix) / 'DATA/data/aerialvln/{}.json'.format(split)), 'r', encoding='utf-8') as f:
			new_data = json.load(f)
			# vocab = new_data['instruction_vocab']
			new_data = new_data['episodes']

		# Partition
		if number > 0:
			random.seed(1)              # Make the data deterministic, additive
			random.shuffle(new_data)    # Shuffle the data
			new_data = new_data[:number]

		# Join
		data += new_data
	random.setstate(old_state)      # Recover the state of the random generator
	return data, vocab


class AirVLNENV:
	"""
	AirVLN 环境类，封装了数据加载、环境交互、状态观测、动作执行等功能。
	主要方法:
	 - __init__: 初始化环境，加载数据，设置动作/观测空间
	 - reset: 重置环境到新的一组 episode
	 - get_obs: 获取当前环境观测
	 - makeActions: 执行动作并更新环境状态
	 - update_measurements: 更新评估指标
	"""

	def __init__(self, batch_size=8, split='train',
				 seed=1, tokenizer=None,
				 dataset_group_by_scene=True,
				 ):
		"""
		环境初始化，加载数据、构建动作/观测空间、初始化 LMDB（根据运行/采集类型）、并建立 VectorEnv 工具。
		关键点:
		 - 处理 tokenization（BERT 或自定义）
		 - 支持不同 collect_type 的 LMDB 配置（TF / dagger / SF）
		 - 支持按场景分组数据集以便机器分配
		参数:
		 - batch_size, split, seed, tokenizer, dataset_group_by_scene
		"""
		self.batch_size = batch_size
		self.split = split
		self.seed = seed
		if tokenizer:
			self.tok = tokenizer
		self.dataset_group_by_scene = dataset_group_by_scene

		load_data, vocab = load_my_datasets([split])
		self.ori_raw_data = load_data.copy()
		self.vocab = vocab.copy() #现阶段是空
		# args.vocab_size = self.vocab['num_vocab']
		logger.info('Loaded with {} instructions, using split: {}'.format(len(load_data), split))

		# 下面这部分是将数据集里面所有数据的instruction_text进行tokenizer处理，得到instruction_tokens，保存在self.data里面
		self.index_data = 0
		self.data = []
		pbar = tqdm.tqdm(total=len(self.ori_raw_data))
		for i_item, item in enumerate(self.ori_raw_data):
			if args.collect_type in ['TF']:
				if len(list(args.TF_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.TF_mode_load_scene):
					pbar.update()
					continue

			if args.collect_type in ['dagger', 'SF']:
				if len(list(args.dagger_mode_load_scene)) > 0 and str(item['scene_id']) not in list(args.dagger_mode_load_scene):
					pbar.update()
					continue

			# 把当前 item 的 instruction 经过 tokenizer 后存入 new_item 并加入 self.data
			new_item = dict(item).copy()
			if args.tokenizer_use_bert:
				text = item['instruction']['instruction_text']
				instruction_tokens = tokenizer(
					text,
					truncation=True,
					max_length=args.maxInput,
					padding='max_length',
					return_tensors="pt"
				)['input_ids'][0]
			else:
				instruction_tokens = tokenizer.encode_sentence(item['instruction']['instruction_text'])
			new_item['instruction']['instruction_tokens'] = instruction_tokens
			self.data.append(new_item)
			pbar.update()
		pbar.close()


		# 构建轨迹id到 instruction_tokens 和 episode_ids 的映射
		# 构建快速映射，会在train.py 里面用到
		self.trajectory_id_2_instruction_tokens = {}
		self.trajectory_id_2_episode_ids = {}
		# 从 self.data 里面提取
		for i_item, item in enumerate(self.data):
			#如果当前轨迹id不在字典里，就新建一个列表并添加instruction_tokens，否则直接添加
			if item['trajectory_id'] not in self.trajectory_id_2_instruction_tokens.keys():
				self.trajectory_id_2_instruction_tokens[item['trajectory_id']] = []
				self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(
					item['instruction']['instruction_tokens']
				)
			else:
				self.trajectory_id_2_instruction_tokens[item['trajectory_id']].append(
					item['instruction']['instruction_tokens']
				)

			if item['trajectory_id'] not in self.trajectory_id_2_episode_ids.keys():
				self.trajectory_id_2_episode_ids[item['trajectory_id']] = []
				self.trajectory_id_2_episode_ids[item['trajectory_id']].append(
					item['episode_id']
				)
			else:
				self.trajectory_id_2_episode_ids[item['trajectory_id']].append(
					item['episode_id']
				)
		#这个结束后，self.trajectory_id_2_instruction_tokens 就是一个字典，key是轨迹id，value是该轨迹下所有episode的instruction_tokens列表
		# 一般一个轨迹只对应一个场景
		# trajectory_id_2_instruction_tokens三维数据结构举例：
		# {
		# 'traj_001': [ [token_list_episode1] ],
		# 'traj_002': [ [token_list_episode2] ],
		# }
		# trajectory_id_2_episode_ids 三维数据结构举例：
		# {
		# 'traj_001': [ episode_id1 ],
		# 'traj_002': [ episode_id2 ],
		# }

		# 打乱数据集顺序
		random.shuffle(self.data)
		# 如果设置了 EVAL_NUM，则多次打乱后取前 EVAL_NUM 条，方便进行评估
		if args.EVAL_NUM != -1 and int(args.EVAL_NUM) > 0:
		 # 多次打乱后取前 EVAL_NUM 条
			[random.shuffle(self.data) for i in range(10)]
			self.data = self.data[:int(args.EVAL_NUM)].copy()
		# 如果设置了 dataset_group_by_scene，则按场景分组，机器就不用切换场景了
		if dataset_group_by_scene:
			self.data = self._group_scenes()
			logger.warning('dataset grouped by scene')

		# 提取场景集合
		scenes = [item['scene_id'] for item in self.data]
		self.scenes = set(scenes)

		# 定义观测空间和动作空间
		self.observation_space = spaces.Dict({
			"rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
			"depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
			"instruction": spaces.Discrete(0),
			"progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32), # 任务进度百分比
			"teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
		})
		self.action_space = spaces.Discrete(int(len(AirsimActions)))

		self.sim_states: Optional[List[SimState], List[None]] = [None for _ in range(batch_size)]
		self.last_scene_id_list = []
		self.one_scene_could_use_num = 5000
		self.this_scene_used_cnt = 0

		# LMDB 初始化，根据不同的 collect_type 和 run_type 设置不同的 LMDB 环境（暂时可以关掉）
		'''
		if args.collect_type in ['TF']:

			if args.run_type in ['collect']:
				# TF collect 模式下，初始化 features、rgb、depth 三个 lmdb
				self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))
				self.lmdb_rgb_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split)+'_rgb'))
				self.lmdb_depth_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / (str(split)+'_depth'))

				if not os.path.exists(str(self.lmdb_features_dir)):
					os.makedirs(str(self.lmdb_features_dir), exist_ok=True)
				if not os.path.exists(str(self.lmdb_rgb_dir)):
					os.makedirs(str(self.lmdb_rgb_dir), exist_ok=True)
				if not os.path.exists(str(self.lmdb_depth_dir)):
					os.makedirs(str(self.lmdb_depth_dir), exist_ok=True)

				lmdb_features_map_size = 5.0e12  # 1.0e11  100GB
				lmdb_rgb_map_size = 5.0e12  # 1.0e11  100GB
				lmdb_depth_map_size = 5.0e12  # 1.0e11  100GB

				try:
					self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
					self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
					self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
					self.threading_lock_lmdb_features_txn = threading.Lock()
					logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

					self.lmdb_collected_keys = set()
					with tqdm.tqdm(
						total=int(self.lmdb_features_start_id), dynamic_ncols=True
					) as pbar:
						for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
							pbar.update()
							self.lmdb_collected_keys.add(key.decode())

					self.lmdb_rgb_env = lmdb.open(self.lmdb_rgb_dir, map_size=int(lmdb_rgb_map_size), readahead=False,)
					self.lmdb_rgb_start_id = self.lmdb_rgb_env.stat()["entries"]
					self.lmdb_rgb_txn = self.lmdb_rgb_env.begin(write=True)
					self.threading_lock_lmdb_rgb_txn = threading.Lock()
					logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'rgb', self.lmdb_rgb_start_id))

					self.lmdb_depth_env = lmdb.open(self.lmdb_depth_dir, map_size=int(lmdb_depth_map_size), readahead=False,)
					self.lmdb_depth_start_id = self.lmdb_depth_env.stat()["entries"]
					self.lmdb_depth_txn = self.lmdb_depth_env.begin(write=True)
					self.threading_lock_lmdb_depth_txn = threading.Lock()
					logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'depth', self.lmdb_depth_start_id))
				except lmdb.Error as err:
					logger.error(err)
					raise err

			if args.run_type in ['eval']:
				self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / '{}_{}'.format(str(split), args.make_dir_time))

				if not os.path.exists(str(self.lmdb_features_dir)):
					os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

				lmdb_features_map_size = 1.0e11  # 1.0e6  1M

				try:
					self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
					self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
					self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
					self.threading_lock_lmdb_features_txn = threading.Lock()
					logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

					self.lmdb_collected_keys = set()
					with tqdm.tqdm(
						total=int(self.lmdb_features_start_id), dynamic_ncols=True
					) as pbar:
						for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
							pbar.update()
							self.lmdb_collected_keys.add(key.decode())

				except lmdb.Error as err:
					logger.error(err)
					raise err

		if args.collect_type in ['dagger', 'SF']:
			self.lmdb_features_dir = str(Path(args.project_prefix) / 'DATA' / 'img_features' / str(args.run_type) / str(args.name) / str(split))

			if not os.path.exists(str(self.lmdb_features_dir)):
				os.makedirs(str(self.lmdb_features_dir), exist_ok=True)

			lmdb_features_map_size = 20.0e12  # 1.0e11  100GB

			try:
				self.lmdb_features_env = lmdb.open(self.lmdb_features_dir, map_size=int(lmdb_features_map_size), readahead=False,)
				self.lmdb_features_start_id = self.lmdb_features_env.stat()["entries"]
				self.lmdb_features_txn = self.lmdb_features_env.begin(write=True)
				self.threading_lock_lmdb_features_txn = threading.Lock()
				logger.info('init lmdb of {}, {}, lmdb_start_id: {}'.format(split, 'features', self.lmdb_features_start_id))

				self.lmdb_collected_keys = set()
				with tqdm.tqdm(
					total=int(self.lmdb_features_start_id), dynamic_ncols=True
				) as pbar:
					for key in self.lmdb_features_txn.cursor().iternext(keys=True, values=False):
						pbar.update()
						if len(str(key.decode()).split('_')) <= 1:
							self.lmdb_collected_keys.add(
								'{}_0'.format(key.decode())
							)
						else:
							self.lmdb_collected_keys.add(key.decode())

			except lmdb.Error as err:
				logger.error(err)
				raise err
		'''

		
		self.init_VectorEnvUtil()

	def _group_scenes(self):
		"""
		按场景对数据进行分组排序，保证同场景的数据连续。
		返回按场景顺序排序后的 self.data（不修改原数据，只重排）。
		"""
		assert self.dataset_group_by_scene, 'error args param'

		scene_sort_keys: Dict[str, int] = {}
		for item in self.data:
			if str(item['scene_id']) not in scene_sort_keys:
				scene_sort_keys[str(item['scene_id'])] = len(scene_sort_keys)

		return sorted(self.data, key=lambda e: scene_sort_keys[str(e['scene_id'])])

	def init_VectorEnvUtil(self):
		"""
		重新初始化 VectorEnvUtil：删除旧实例并用当前场景集创建新的 VectorEnvUtil 以供批量环境交互。
		"""
		self.delete_VectorEnvUtil()

		self.load_scenes = [int(_scene) for _scene in list(self.scenes)]
		# 假设self.scenes是[3,2,5,4,1,11]
		# 那么 self.load_scenes 就是 [3,2,5,4,1,11] 的整数列表形式(其实已经是整数了)
		self.VectorEnvUtil = VectorEnvUtil(self.load_scenes, self.batch_size)

	def delete_VectorEnvUtil(self):
		"""
		删除现有的 VectorEnvUtil 实例并尝试触发垃圾回收，释放相关资源。
		"""
		if hasattr(self, 'VectorEnvUtil'):
			del self.VectorEnvUtil

		import gc
		gc.collect()

	def next_minibatch(self, skip_scenes=[], data_it=0):
		"""
		从数据集中构造下一个 minibatch：
		 - 跳过指定 skip_scenes
		 - 根据 lmdb 已收集键避免重复采集
		 - 支持按 collect_type (TF / dagger / SF) 的键策略
		成功后调用 VectorEnvUtil.set_batch 设置当前批次。
		参数:
		 - skip_scenes: 要跳过的场景 id 列表
		 - data_it: dagger/SF 使用的迭代索引，用于构造 lmdb key
		"""
		batch = []

		while True:
			#如果index_data 超过数据集长度，说明已经遍历完一轮数据集了，那就打乱数据集重新开始
			# 如果当前 batch 为空，说明没有数据可取，直接返回
			if self.index_data >= len(self.data)-1:
				random.shuffle(self.data)
				logger.warning('random shuffle data')
				if self.dataset_group_by_scene:
					self.data = self._group_scenes()
					logger.warning('dataset grouped by scene')

				if len(batch) == 0:
					self.index_data = 0
					self.batch = None
					return

				self.index_data = self.batch_size - len(batch)
				batch += self.data[:self.index_data]
				break

			# 正常情况走这里，用 index_data 表明当前到数据集的哪个位置了
			new_episode = self.data[self.index_data]

			# 跳过指定场景
			if new_episode['scene_id'] in skip_scenes:
				self.index_data += 1
				continue

			# 避免重复采集
			'''
			if args.run_type in ['collect', 'train'] and args.collect_type in ['TF']:
				lmdb_key = '{}'.format(new_episode['episode_id'])
				# 如果 lmdb_key 已存在，说明已经采集过了，跳过
				if lmdb_key in self.lmdb_collected_keys:
					self.index_data += 1
					continue
				else:
					batch.append(new_episode)
					self.index_data += 1
			elif args.run_type in ['collect', 'train'] and args.collect_type in ['dagger', 'SF']:
				lmdb_key = '{}_{}'.format(new_episode['episode_id'], data_it)
				if lmdb_key in self.lmdb_collected_keys:
					self.index_data += 1
					continue
				else:
					batch.append(new_episode)
					self.index_data += 1
			else:
				batch.append(new_episode)
				self.index_data += 1
			'''
			# 简化版,不考虑 lmdb 重复采集问题
			batch.append(new_episode)
			self.index_data += 1
			
			if len(batch) == self.batch_size:
				break

		self.batch = copy.deepcopy(batch)
		#self.batch 现在是一个长度为 batch_size 的 episode 列表，是下一批次的数据
		assert len(self.batch) == self.batch_size, 'next_minibatch error'

		# 设置当前批次到 VectorEnvUtil
		self.VectorEnvUtil.set_batch(self.batch)


	def changeToNewEpisodes(self):
		"""
		切换到新的一组 episode：
		 - 可能会触发环境切换（_changeEnv）
		 - 设置 episode 初始位姿（_setEpisodes）
		 - 更新各类测量指标（update_measurements）
		"""
		self._changeEnv(need_change=False)

		self._setEpisodes()

		self.update_measurements()

	def _changeEnv(self, need_change: bool = True):
		"""
		根据当前批次分配机器(open_scenes)，在需要时与 AirSim 实例建立或切换连接。
		参数:
		 - need_change: 强制切换标志（True 则强制重置场景）
		要点:
		 - 根据 args.machines_info 配置把场景分配到不同机器
		 - 重试直至成功启动 simulator tool
		"""
		#从self.batch 里面提取当前批次的场景 id 列表
		scene_id_list = [item['scene_id'] for item in self.batch]
		assert len(scene_id_list) == self.batch_size, 'error'

		# 计算总的 MAX_SCENE_NUM 并断言 batch_size 不超过总和，一台机器不用管
		machines_info_template = copy.deepcopy(args.machines_info)
		total_max_scene_num = 0
		for item in machines_info_template:
			total_max_scene_num += item['MAX_SCENE_NUM']
		assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

		# 构造机器信息 TODO，这里是多台机器的情况，目前暂时遇不到，不用管
		machines_info = []
		ix = 0
		for index, item in enumerate(machines_info_template):
			machines_info.append(item)
			delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(scene_id_list)-ix)
			machines_info[index]['open_scenes'] = scene_id_list[ix : ix + delta]
			ix += delta

		# 统计每台机器分配到的场景数量，断言总和等于 batch_size
		cnt = 0
		for item in machines_info:
			cnt += len(item['open_scenes'])
		assert self.batch_size == cnt, 'error create machines_info'
		# 上面这些都是为了把当前批次的场景分配到不同机器上去，只有一台机器的话就没啥变化

		# 写死了一个场景最多使用5000次
		# 如果当前场景和上次场景相同，且使用次数没超过限制，且不需要强制切换，就不切换环境
		# 当前场景使用次数加1后直接返回
		if self.this_scene_used_cnt < self.one_scene_could_use_num and \
				len(set(scene_id_list)) == 1 and len(set(self.last_scene_id_list)) == 1 and \
				scene_id_list[0] is not None and self.last_scene_id_list[0] is not None and scene_id_list[0] == self.last_scene_id_list[0] and \
				need_change == False:
			self.this_scene_used_cnt += 1
			logger.warning('no need to change env: {}'.format(scene_id_list))
			return
		else:
			logger.warning('to change env: {}'.format(scene_id_list))

		# 循环尝试启动 simulator tool 直至成功，调用 AirVLNSimulatorClientTool
		# 如果有异常就等待3秒后重试
		while True:
			try:
				self.machines_info = copy.deepcopy(machines_info)
				if (not args.ablate_rgb or not args.ablate_depth):
					self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
					self.simulator_tool.run_call()
				break
			except Exception as e:
				logger.error("Failed to open scenes {}".format(e))
				time.sleep(3)
			except:
				logger.error('Failed to open scenes')
				time.sleep(3)

		# 记录当前场景 id 列表，重置使用计数
		self.last_scene_id_list = scene_id_list.copy()
		self.this_scene_used_cnt = 1

	def _setEpisodes(self):
		"""
		将当前批次的起始位姿设到 simulator，并在 sim_states 中初始化每个实例的 SimState（位姿、轨迹等）。
		如果设置失败会尝试 reset_to_this_pose 进行重设。
		"""
		start_position_list = [item['start_position'] for item in self.batch]
		start_rotation_list = [item['start_rotation'] for item in self.batch]

		#计算并构造 poses 列表
		poses = []
		cnt = 0
		for index_1, item in enumerate(self.machines_info):
			poses.append([])
			for index_2, _ in enumerate(item['open_scenes']):
				pose = airsim.Pose(
					position_val=airsim.Vector3r(
						x_val=start_position_list[cnt][0],
						y_val=start_position_list[cnt][1],
						z_val=start_position_list[cnt][2],
					),
					orientation_val=airsim.Quaternionr(
						x_val=start_rotation_list[cnt][1],
						y_val=start_rotation_list[cnt][2],
						z_val=start_rotation_list[cnt][3],
						w_val=start_rotation_list[cnt][0],
					),
				)
				poses[index_1].append(pose)
				cnt += 1

		# 发送位姿到 simulator
		if (not args.ablate_rgb or not args.ablate_depth):
			result = self.simulator_tool.setPoses(poses=poses)
			if not result:
				logger.error('Failed to set poses')
				self.reset_to_this_pose(poses)

		# 依次初始化模拟器状态
		cnt = 0
		for index_1, item in enumerate(self.machines_info):
			for index_2, _ in enumerate(item['open_scenes']):
				pose = airsim.Pose(
					position_val=airsim.Vector3r(
						x_val=start_position_list[cnt][0],
						y_val=start_position_list[cnt][1],
						z_val=start_position_list[cnt][2],
					),
					orientation_val=airsim.Quaternionr(
						x_val=start_rotation_list[cnt][1],
						y_val=start_rotation_list[cnt][2],
						z_val=start_rotation_list[cnt][3],
						w_val=start_rotation_list[cnt][0],
					),
				)
				self.sim_states[cnt] = SimState(index=cnt, step=0, episode_info=self.batch[cnt], pose=pose)
				self.sim_states[cnt].trajectory = [[
					pose.position.x_val, pose.position.y_val, pose.position.z_val, # xyz
					pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val, # xyzw
				]]
				cnt += 1

	# 获取当前环境观测（所有），放在self.sim_states里面
	def get_obs(self):
		"""
		获取当前环境观测：
		 - 从 simulator 获取图像/深度（通过 _getStates）
		 - 调用 VectorEnvUtil.get_obs 返回格式化的观测和新状态
		返回:
		 - obs: 格式化的观测字典/批次
		"""
		obs_states = self._getStates()

		obs, states = self.VectorEnvUtil.get_obs(obs_states)
		self.sim_states = states

		return obs

	def _getStates(self):
		"""
		向 simulator 请求原始图像/深度响应并构造成内部状态列表：
		 - 检测并处理 collision（基于深度）
		 - 将原始图像保存到 lmdb（在对应的采集模式下）
		 - 返回 states 列表，每项为 (rgb_array_or_None, depth_array_or_None, SimState)
		"""
		while True:
			if (not args.ablate_rgb or not args.ablate_depth):
				# 从 simulator 获取图像响应
				responses = self.simulator_tool.getImageResponses(get_rgb=not bool(args.ablate_rgb), get_depth=not bool(args.ablate_depth))
			else:
				# 如果是消融实验，直接返回空响应
				responses = [[(None, None) for j in range(self.batch_size)] for i in range(len(self.machines_info))]
			# 如果获取失败，尝试 reset_to_this_pose 后重试
			if responses is None:
				poses = self._get_current_pose()
				self.reset_to_this_pose(poses)
				time.sleep(3)
			else:
				break

		# 检查responses个数是否正确
		cnt = 0
		for item in responses:
			cnt += len(item)
		assert len(responses) == len(self.machines_info), 'error'
		assert cnt == self.batch_size, 'error'

		# 处理 collision 检测
		# 如果是 eval 模式，或者 collect 模式下的 dagger，则进行 collision 检测
		if args.run_type in ['eval'] or \
			(args.run_type in ['collect'] and args.collect_type in ['dagger']):
			cnt = 0
			for index_1, item in enumerate(self.machines_info):
				for index_2 in range(len(item['open_scenes'])):
					# 打开深度图像进行碰撞检测
					depth_image = responses[index_1][index_2][1]
					collision_sensor_result = (np.array(depth_image) < 0.004).sum() / np.array(depth_image).flatten().shape[0]
					# 如果碰撞比例超过阈值，标记为碰撞并结束
					if collision_sensor_result > 0.1:
						self.sim_states[cnt].is_collisioned = True
						self.sim_states[cnt].is_end = True
						logger.warning('collisioned: {}'.format(cnt))

					cnt += 1

		# 构造 states 列表
		states = [None for _ in range(self.batch_size)]
		cnt = 0
		for index_1, item in enumerate(self.machines_info):
			for index_2 in range(len(item['open_scenes'])):
				rgb_image = responses[index_1][index_2][0]
				if rgb_image is not None:
					_rgb_image = np.array(rgb_image)
				else:
					_rgb_image = None

				depth_image = responses[index_1][index_2][1]
				if depth_image is not None:
					_depth_image = np.array(depth_image)
				else:
					_depth_image = None

				state = self.sim_states[cnt]

				# 每次states，存下rgb图像、depth图像和state
				states[cnt] = (_rgb_image, _depth_image, state)
				cnt += 1

				# 如果是train collect TF模式，就把图像存到 lmdb 里面
				'''
				if self.split in ['train'] and args.run_type in ['collect'] and args.collect_type in ['TF']:
					trajectory_id = state.episode_info['trajectory_id']
					step = state.step
					lmdb_rgb_key = '{}_{}_rgb'.format(trajectory_id, step)
					lmdb_depth_key = '{}_{}_depth'.format(trajectory_id, step)

					if rgb_image is not None:
						self.threading_lock_lmdb_rgb_txn.acquire()
						self.lmdb_rgb_txn.put(
							lmdb_rgb_key.encode(),
							msgpack_numpy.packb(
								rgb_image, use_bin_type=True
							),
						)
						self.threading_lock_lmdb_rgb_txn.release()

					if depth_image is not None:
						self.threading_lock_lmdb_depth_txn.acquire()
						self.lmdb_depth_txn.put(
							lmdb_depth_key.encode(),
							msgpack_numpy.packb(
								depth_image, use_bin_type=True
							),
						)
						self.threading_lock_lmdb_depth_txn.release()
				'''
		return states

	def _get_current_pose(self) -> list:
		"""
		从 self.sim_states 中读取当前每个实例的 pose，并按 machines_info 的分布返回嵌套列表（与 setPoses 接口一致）。
		返回:
		 - poses: list[list[airsim.Pose]]，外层每项对应一台机器
		"""
		poses = []

		cnt = 0
		for index_1, item in enumerate(self.machines_info):
			poses.append([])
			for index_2, _ in enumerate(item['open_scenes']):
				poses[index_1].append(
					self.sim_states[cnt].pose
				)
				cnt += 1

		return poses


	def reset(self):
		"""
		重置环境到新 episode 并返回初始观测（等同于 changeToNewEpisodes + get_obs）。
		目前这个函数并不会被外部调用到，因为外部调用的是 VectorEnvUtil 里面的 reset。
		"""
		self.changeToNewEpisodes()
		return self.get_obs()


	def reset_to_this_pose(self, poses):
		"""
		强制切换环境并将所有实例设置到指定 poses（用于出错恢复）。
		参数:
		 - poses: 与 setPoses 接口匹配的嵌套位姿列表
		实现: 若设置失败会递归尝试直至成功（与原实现保持一致）。
		"""
		# 强制切换环境
		self._changeEnv(need_change=True)

		# 设置位姿
		if (not args.ablate_rgb or not args.ablate_depth):
			result = self.simulator_tool.setPoses(poses=poses)
			if not result:
				logger.error('Failed to reset to this pose')
				self.reset_to_this_pose(poses)


	def makeActions(self, action_list):
		"""
		对批次中的每个实例执行动作列表：
		 - 计算新位姿（getPoseAfterMakeAction）
		 - 将新位姿发送给 simulator（按机器分组）
		 - 更新 sim_states（step、pose、trajectory、is_end、pre_action）
		 - 非 collect 模式下会触发 update_measurements
		参数:
		 - action_list: 长度为 batch_size 的动作枚举列表
		"""
		#依次执行每个动作，计算新位姿
		poses = []
		for index, action in enumerate(action_list):
			if self.sim_states[index].is_end == True:
				action = AirsimActions.STOP
				# continue

			if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
				self.sim_states[index].is_end = True


			state = self.sim_states[index]

			pose = copy.deepcopy(state.pose)
			#将动作作用到当前位姿，得到新位姿
			#这里是关键代码！！！
			new_pose = getPoseAfterMakeAction(pose, action)
			poses.append(new_pose)

		# 格式化 poses 列表以匹配 setPoses 接口
		poses_formatted = []
		cnt = 0
		for index_1, item in enumerate(self.machines_info):
			poses_formatted.append([])
			for index_2, _ in enumerate(item['open_scenes']):
				poses_formatted[index_1].append(poses[cnt])
				cnt += 1

		# 发送位姿给 simulator
		if (not args.ablate_rgb or not args.ablate_depth):
			result = self.simulator_tool.setPoses(poses=poses_formatted)
			if not result:
				logger.error('Failed to set poses')
				self.reset_to_this_pose(poses_formatted)

		# 更新 sim_states
		for index, action in enumerate(action_list):
			if self.sim_states[index].is_end == True:
				continue

			if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
				self.sim_states[index].is_end = True

			self.sim_states[index].step += 1
			self.sim_states[index].pose = poses[index]
			self.sim_states[index].trajectory.append([
				poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
				poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val, # xyzw
			])
			self.sim_states[index].pre_action = action

		# update measurement
		if args.run_type not in ['collect']:
			self.update_measurements()


	#
	def update_measurements(self):
		"""
		更新所有评估指标的包装方法，会依次调用：
		 - _update_DistanceToGoal
		 - _updata_Success
		 - _updata_NDTW
		 - _updata_SDTW
		 - _update_PathLength
		 - _update_OracleSuccess
		 - _update_StepsTaken
		"""
		self._update_DistanceToGoal()
		self._updata_Success()
		self._updata_NDTW()
		self._updata_SDTW()
		self._update_PathLength()
		self._update_OracleSuccess()
		self._update_StepsTaken()

	def _update_DistanceToGoal(self):
		"""
		计算并更新每个实例到目标位置的当前距离（基于 XY 平面欧氏距离），并缓存 previous_position 以避免重复计算。
		"""
		for i, state in enumerate(self.sim_states):

			current_position = np.array([
				state.pose.position.x_val,
				state.pose.position.y_val,
				state.pose.position.z_val
			])

			if self.sim_states[i].DistanceToGoal['_previous_position'] is None or \
				not np.allclose(self.sim_states[i].DistanceToGoal['_previous_position'], current_position, atol=1):
				distance_to_target = EuclideanDistance3(
					np.array(current_position)[0:2],
					np.array(state.episode_info['goals'][0]['position'])[0:2]
				)
				self.sim_states[i].DistanceToGoal['_previous_position'] = current_position
				self.sim_states[i].DistanceToGoal['_metric'] = distance_to_target

	def _updata_Success(self):
		"""
		根据 is_end 与 DistanceToGoal 判断当前 episode 是否成功（Success metric 为 0 或 1）。
		"""
		for i, state in enumerate(self.sim_states):
			distance_to_target = self.sim_states[i].DistanceToGoal['_metric']
			if (
				self.sim_states[i].is_end
				and distance_to_target <= self.sim_states[i].SUCCESS_DISTANCE
			):
				self.sim_states[i].Success['_metric'] = 1.0
			else:
				self.sim_states[i].Success['_metric'] = 0.0

	def _updata_NDTW(self):
		"""
		使用 fastdtw 计算当前轨迹与参考轨迹之间的归一化 DTW（nDTW）指标并保存到 NDTW['_metric']。
		内部包含一个用于计算两点欧氏距离的辅助函数 euclidean_distance。
		"""
		def euclidean_distance(
				position_a,
				position_b,
		) -> float:
			"""
			计算两点间的欧氏距离（2范数）。
			位置可为可迭代数值序列。
			"""
			return np.linalg.norm(
				np.array(position_b) - np.array(position_a), ord=2
			)

		for i, state in enumerate(self.sim_states):

			current_position = np.array([
				state.pose.position.x_val,
				state.pose.position.y_val,
				state.pose.position.z_val
			])

			if len(state.NDTW['locations']) == 0:
				self.sim_states[i].NDTW['locations'].append(current_position)
			else:
				if current_position.tolist() == state.NDTW['locations'][-1].tolist():
					continue
				self.sim_states[i].NDTW['locations'].append(current_position)

			dtw_distance = fastdtw(
				self.sim_states[i].NDTW['locations'], self.sim_states[i].NDTW['gt_locations'], dist=euclidean_distance
			)[0]

			nDTW = np.exp(
				-dtw_distance / (len(self.sim_states[i].NDTW['gt_locations']) * self.sim_states[i].SUCCESS_DISTANCE)
			)
			self.sim_states[i].NDTW['_metric'] = nDTW

	def _updata_SDTW(self):
		"""
		计算并更新 SDTW（Success * nDTW），将结果写入 SDTW['_metric']。
		"""
		for i, state in enumerate(self.sim_states):
			ep_success = self.sim_states[i].Success['_metric']
			nDTW = self.sim_states[i].NDTW['_metric']
			self.sim_states[i].SDTW['_metric'] = ep_success * nDTW

	def _update_PathLength(self):
		"""
		累积并更新路径长度（PathLength['_metric']），基于每一步位置增量计算并记录 previous_position。
		"""
		for i, state in enumerate(self.sim_states):

			current_position = np.array([
				state.pose.position.x_val,
				state.pose.position.y_val,
				state.pose.position.z_val
			])

			if state.PathLength['_previous_position'] is None:
				self.sim_states[i].PathLength['_previous_position'] = current_position

			self.sim_states[i].PathLength['_metric'] += EuclideanDistance3(
				current_position, self.sim_states[i].PathLength['_previous_position']
			)
			self.sim_states[i].PathLength['_previous_position'] = current_position

	def _update_OracleSuccess(self):
		"""
		更新 OracleSuccess 指标：当历史上任一步距离小于 SUCCESS_DISTANCE 则为 True（转为 float）。
		"""
		for i, state in enumerate(self.sim_states):
			d = self.sim_states[i].DistanceToGoal['_metric']
			self.sim_states[i].OracleSuccess['_metric'] = float(
				self.sim_states[i].OracleSuccess['_metric'] or d <= self.sim_states[i].SUCCESS_DISTANCE
			)

	def _update_StepsTaken(self):
		"""
		更新 StepsTaken 指标为当前步骤计数（sim_state.step）。
		"""
		for i, state in enumerate(self.sim_states):
			self.sim_states[i].StepsTaken['_metric'] = self.sim_states[i].step

