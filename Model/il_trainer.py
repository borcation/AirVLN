import os
from typing import Dict

import torch
import torch.nn.functional as F
from gym import Space
from torch.cuda.amp import autocast, GradScaler

from Model.seq2seq_policy import Seq2SeqPolicy
from Model.cma_policy import CMAPolicy
from utils.logger import logger
from src.common.param import args
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN


class VLNCETrainer:
    """
    智能体训练器，负责模型的初始化、训练、保存、加载等核心功能。
    """
    def __init__(
        self,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        ckpt_path=None,
    ):
        # 训练起始 epoch 和 step 计数
        self.start_epoch = 0
        self.step_id = 0

        # 设备选择：支持单卡和分布式训练
        if not args.DistributedDataParallel:
            self.device = (
                torch.device("cuda", args.trainer_gpu_device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = (
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        
        logger.info(f"Trainer initialized on device: {self.device}")

        # 克隆模型配置
        model_config = CN.clone()
        # 根据 policy_type 选择不同的策略模型
        if args.policy_type == 'seq2seq':
            self.policy = Seq2SeqPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        elif args.policy_type == 'cma':
            self.policy = CMAPolicy.from_config(
                observation_space=observation_space,
                action_space=action_space,
                out_model_config=model_config,
                device=self.device,
            )
        else:
            raise NotImplementedError

        # 将模型放到指定设备
        self.policy.to(self.device)

        # 优化器，使用 Adam
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=args.lr
        )

        # AMP 混合精度（可选）
        self.use_amp = bool(getattr(args, "amp", False)) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # 如果需要从 checkpoint 加载参数
        # 类似使用checkpoint这种方式来进行重连恢复的
        if load_from_ckpt:
            assert os.path.isfile(ckpt_path), 'ckpt_path error'
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.policy.load_state_dict(ckpt_dict["state_dict"])
            self.optimizer.load_state_dict(ckpt_dict["optimizer"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

        # 分布式训练包装
        if args.DistributedDataParallel:
            self.policy = torch.nn.parallel.DistributedDataParallel(
                self.policy,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        # 打印模型参数量
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params}. Trainable: {params_t}")
        logger.info("Finished setting up policy.")

    #
    def save_checkpoint(self, file_name, dagger_it, epoch) -> None:
        """
        保存当前模型和优化器的状态到 checkpoint 文件。
        :param file_name: 文件名
        :param dagger_it: 当前 DAgger 迭代次数
        :param epoch: 当前 epoch
        :return: None
        """
        checkpoint = {
            # 分布式和单机保存方式不同
            "state_dict": self.policy.module.state_dict() if args.DistributedDataParallel else self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "config": str(args),
            'dagger_it': dagger_it,
            'epoch': epoch,
        }

        from pathlib import Path
        checkpoint_folder = Path(args.project_prefix) / 'DATA/output/{}/train/checkpoint/{}'.format(args.name, args.make_dir_time)
        if not os.path.exists(str(checkpoint_folder)):
            os.makedirs(str(checkpoint_folder), exist_ok=True)

        torch.save(
            checkpoint, str(checkpoint_folder / file_name)
        )

    #
    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        """
        加载 checkpoint 文件，返回内容字典。
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    #
    def _update_agent(
        self,
        observations,
        prev_actions,
        not_done_masks,
        corrected_actions,
        weights,
        step_grad: bool = True,
        loss_accumulation_scalar: int = 1,
    ):
        """
        用一批数据对智能体进行一次参数更新。
        :param observations: 当前观测
        :param prev_actions: 上一步动作
        :param not_done_masks: 未结束标志
        :param corrected_actions: 正确动作（监督信号）
        :param weights: 损失权重
        :param step_grad: 是否执行梯度更新
        :param loss_accumulation_scalar: 梯度累积因子
        :return: 总损失、动作损失、辅助损失
        """
        T, N = corrected_actions.size()  # T: 序列长度，N: batch size

        # 初始化 RNN 隐状态
        if args.policy_type in ['seq2seq', 'cma']:
            if not args.DistributedDataParallel:
                recurrent_hidden_states = torch.zeros(
                    N,
                    self.policy.net.num_recurrent_layers,
                    self.policy.net.state_encoder.hidden_size,
                    device=self.device,
                )
            else:
                recurrent_hidden_states = torch.zeros(
                    N,
                    self.policy.module.net.num_recurrent_layers,
                    self.policy.module.net.state_encoder.hidden_size,
                    device=self.device,
                )
        else:
            raise NotImplementedError

        # 清空辅助损失缓存
        AuxLosses.clear()

        # 前向传播与损失（支持 AMP）
        if self.use_amp:
            with autocast():
                if not args.DistributedDataParallel:
                    distribution = self.policy.build_distribution(
                        observations, recurrent_hidden_states, prev_actions, not_done_masks
                    )
                else:
                    distribution = self.policy.module.build_distribution(
                        observations, recurrent_hidden_states, prev_actions, not_done_masks
                    )

                logits = distribution.logits.view(T, N, -1)
                action_loss = F.cross_entropy(
                    logits.permute(0, 2, 1), corrected_actions, reduction="none"
                )
                action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

                aux_mask = (weights > 0).view(-1)
                aux_loss = AuxLosses.reduce(aux_mask)

                loss = (action_loss + aux_loss) / loss_accumulation_scalar

            self.scaler.scale(loss).backward()
        else:
            if not args.DistributedDataParallel:
                distribution = self.policy.build_distribution(
                    observations, recurrent_hidden_states, prev_actions, not_done_masks
                )
            else:
                distribution = self.policy.module.build_distribution(
                    observations, recurrent_hidden_states, prev_actions, not_done_masks
                )

            logits = distribution.logits.view(T, N, -1)
            action_loss = F.cross_entropy(
                logits.permute(0, 2, 1), corrected_actions, reduction="none"
            )
            action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

            aux_mask = (weights > 0).view(-1)
            aux_loss = AuxLosses.reduce(aux_mask)

            loss = (action_loss + aux_loss) / loss_accumulation_scalar
            loss.backward()

        # 梯度更新
        if step_grad:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if isinstance(aux_loss, torch.Tensor):
            aux_loss = aux_loss.item()
        return loss.item(), action_loss.item(), aux_loss

