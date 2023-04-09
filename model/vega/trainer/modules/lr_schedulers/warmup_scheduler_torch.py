# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Basic Warm up lr scheduler.

Example:
    >>> # in yml file `trainer` part
    >>> # use WarmupScheduler and MultiStepLR as after_scheduler
    >>> lr_scheduler:
    >>>     type: WarmupScheduler
    >>>     by_epoch: False
    >>>     params:
    >>>         warmup_type: linear | constant | exp
    >>>         warmup_iters: 20
    >>>         warmup_ratio: 0.1
    >>>         after_scheduler_config:
    >>>             by_epoch: False
    >>>             type: MultiStepLR
    >>>             params:
    >>>                 milestones: [60, 120]
    >>>                 gamma: 0.5

"""

from vega.common import ClassFactory, ClassType
from torch.optim.lr_scheduler import _LRScheduler
from vega.trainer.modules.lr_schedulers import LrScheduler

import torch

@ClassFactory.register(ClassType.LR_SCHEDULER)
class WarmupScheduler(_LRScheduler):
    """Multiple Step learning rate with warm up.

    :param milestones: list of decay epochs
    :type milestones: list of int
    :param decay_rates: list of decay rates
    :type decay_rates: list of float
    :param warmup: whether to warm up
    :type warmup: bool
    :param epoch_steps: steps in one epoch
    :type epoch_steps: int
    """

    def __init__(self,
                 optimizer,
                 warmup_type='linear',
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 after_scheduler_config=None,
                 **kwargs):
        """Init WarmupScheduler."""
        if warmup_type is not None:
            if not isinstance(warmup_iters, int) or warmup_iters <= 0:
                raise ValueError('"warmup_iters" must be a positive integer')
            if not isinstance(warmup_ratio, float) or warmup_ratio <= 0 or warmup_ratio > 1.0:
                raise ValueError('"warmup_ratio" must be in range (0,1]')
        self.optimizer = optimizer
        self.warmup_type = warmup_type
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.after_scheduler_config = after_scheduler_config
        self.current_iters = 0
        self.warmup_finished = False
        super(WarmupScheduler, self).__init__(optimizer)

        self.lane_first_time = True
        self.seg_first_time = True
        self.det_first_time = True
        self.lr_tuning_init = 0.0

    def get_lr(self):
        """Get lr."""
        if self.warmup_finished:
            return self.after_scheduler.get_lr()

        if self.warmup_type == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lrs]
        elif self.warmup_type == 'linear':
            k = (1 - self.current_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lrs]
        elif self.warmup_type == 'exp':
            k = self.warmup_ratio ** (1 - self.current_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lrs]
        return warmup_lr

    def step(self, epoch=None):
        """Step forward for current scheduler."""

        if self.warmup_finished:
            # BACKBONE_START = 3
            LANE_START = 113
            SEG_START = 144
            DETECT_START = 162
            END = 198

            ITER_PER_EPOCH = 111.0
            joint_train_epoch = 10
            train_lane_alone = 5
            train_det_alone = 5
            train_seg_alone = 5

            epoch_index = int(epoch / ITER_PER_EPOCH) % (joint_train_epoch + train_lane_alone + train_seg_alone + train_det_alone)

            #---------------------------------------------------#
            #  first 只训练车道线分支头
            #---------------------------------------------------#
            if joint_train_epoch + train_lane_alone > epoch_index >= joint_train_epoch:
                if self.lane_first_time:
                    self.lane_first_time = False
                    print("=====================start training lane header alone=====================")

                    #---------------------------------------------------#
                    #  提取通用参数
                    #---------------------------------------------------#
                    lr = self.after_scheduler.optimizer.param_groups[0]["lr"]
                    self.lr_tuning_init = lr

                    momentum = self.after_scheduler.optimizer.param_groups[0]["momentum"]
                    dampening = self.after_scheduler.optimizer.param_groups[0]["dampening"]
                    weight_decay = self.after_scheduler.optimizer.param_groups[0]["weight_decay"]
                    nesterov = self.after_scheduler.optimizer.param_groups[0]["nesterov"]

                    # 参数列表
                    param_list = []
                    for param_dict in self.after_scheduler.optimizer.param_groups:
                        param_list.extend(param_dict["params"])

                    # 三个动态权重清零
                    param_list[0].data = torch.tensor(0.0,device = param_list[0].data.device)
                    param_list[1].data = torch.tensor(0.0,device = param_list[1].data.device)
                    param_list[2].data = torch.tensor(0.0,device = param_list[2].data.device)

                    #---------------------------------------------------#
                    #  车道线 -- 非车道线部分
                    #---------------------------------------------------#
                    # 车道线部分
                    param_lane_dict = dict()
                    param_activate_lane = param_list[LANE_START:SEG_START]
                    param_lane_dict["params"] = param_activate_lane
                    param_lane_dict["lr"] = lr
                    param_lane_dict["momentum"] = momentum
                    param_lane_dict["dampening"] = dampening
                    param_lane_dict["weight_decay"] = weight_decay
                    param_lane_dict["nesterov"] = nesterov
                    param_lane_dict["initial_lr"] = lr

                    # 非车道线部分
                    param_other_dict = {}
                    param_freeze_lane = param_list[0:LANE_START] + param_list[SEG_START:]
                    param_other_dict["params"] = param_freeze_lane
                    param_other_dict["lr"] = 0.00   # 冻结
                    param_other_dict["momentum"] = momentum
                    param_other_dict["dampening"] = dampening
                    param_other_dict["weight_decay"] = weight_decay
                    param_other_dict["nesterov"] = nesterov
                    param_other_dict["initial_lr"] = 0.0

                    self.after_scheduler.optimizer.param_groups[0] = param_lane_dict
                    self.after_scheduler.optimizer.param_groups.append(param_other_dict)

            #---------------------------------------------------#
            #  second 只训练检测分支头
            #---------------------------------------------------#
            if joint_train_epoch + train_lane_alone+ train_det_alone > epoch_index >= joint_train_epoch + train_lane_alone:
                if self.det_first_time:
                    self.det_first_time = False
                    print("=====================start training detection header alone=====================")

                    #---------------------------------------------------#
                    #  提取通用参数
                    #---------------------------------------------------#
                    momentum = self.after_scheduler.optimizer.param_groups[0]["momentum"]
                    dampening = self.after_scheduler.optimizer.param_groups[0]["dampening"]
                    weight_decay = self.after_scheduler.optimizer.param_groups[0]["weight_decay"]
                    nesterov = self.after_scheduler.optimizer.param_groups[0]["nesterov"]

                    # 参数列表
                    param_one = self.after_scheduler.optimizer.param_groups[0]["params"]
                    param_two = self.after_scheduler.optimizer.param_groups[1]["params"]
                    param_list = param_two[0:LANE_START] + param_one + param_two[LANE_START:]

                    #---------------------------------------------------#
                    #  目标检测 -- 非目标部分
                    #---------------------------------------------------#
                    # 检测头
                    param_det_dict = dict()
                    param_activate_det = param_list[DETECT_START:END]
                    param_det_dict["params"] = param_activate_det
                    param_det_dict["lr"] = self.lr_tuning_init
                    param_det_dict["momentum"] = momentum
                    param_det_dict["dampening"] = dampening
                    param_det_dict["weight_decay"] = weight_decay
                    param_det_dict["nesterov"] = nesterov
                    param_det_dict["initial_lr"] = self.lr_tuning_init

                    # 非检测头部分
                    param_freeze_det = {}
                    param_freeze_det_param = param_list[0:DETECT_START] + param_list[END:]
                    param_freeze_det["params"] = param_freeze_det_param
                    param_freeze_det["lr"] = 0.00   # 冻结
                    param_freeze_det["momentum"] = momentum
                    param_freeze_det["dampening"] = dampening
                    param_freeze_det["weight_decay"] = weight_decay
                    param_freeze_det["nesterov"] = nesterov
                    param_freeze_det["initial_lr"] = 0.0

                    self.after_scheduler.optimizer.param_groups[0] = param_det_dict
                    self.after_scheduler.optimizer.param_groups[1] = param_freeze_det

            #---------------------------------------------------#
            #  third 只训练分割分支头
            #---------------------------------------------------#
            if joint_train_epoch + train_lane_alone+ train_det_alone+train_seg_alone > epoch_index >= joint_train_epoch + train_lane_alone+ train_det_alone:
                if self.seg_first_time:
                    self.seg_first_time = False
                    print("=====================start training seg header alone=====================")

                    #---------------------------------------------------#
                    #  提取通用参数
                    #---------------------------------------------------#
                    momentum = self.after_scheduler.optimizer.param_groups[0]["momentum"]
                    dampening = self.after_scheduler.optimizer.param_groups[0]["dampening"]
                    weight_decay = self.after_scheduler.optimizer.param_groups[0]["weight_decay"]
                    nesterov = self.after_scheduler.optimizer.param_groups[0]["nesterov"]

                    # 参数列表
                    param_one = self.after_scheduler.optimizer.param_groups[0]["params"]
                    param_two = self.after_scheduler.optimizer.param_groups[1]["params"]
                    param_list = param_two[0:DETECT_START] + param_one + param_two[DETECT_START:]

                    #---------------------------------------------------#
                    #  语义分割部分
                    #---------------------------------------------------#
                    # 分割头
                    param_det_dict = dict()
                    param_activate_det = param_list[SEG_START:DETECT_START]
                    param_det_dict["params"] = param_activate_det
                    param_det_dict["lr"] = self.lr_tuning_init
                    param_det_dict["momentum"] = momentum
                    param_det_dict["dampening"] = dampening
                    param_det_dict["weight_decay"] = weight_decay
                    param_det_dict["nesterov"] = nesterov
                    param_det_dict["initial_lr"] = self.lr_tuning_init

                    # 非分割头部分
                    param_freeze_det = {}
                    param_freeze_det_param = param_list[0:SEG_START] + param_list[DETECT_START:]
                    param_freeze_det["params"] = param_freeze_det_param
                    param_freeze_det["lr"] = 0.00   # 冻结
                    param_freeze_det["momentum"] = momentum
                    param_freeze_det["dampening"] = dampening
                    param_freeze_det["weight_decay"] = weight_decay
                    param_freeze_det["nesterov"] = nesterov
                    param_freeze_det["initial_lr"] = 0.00

                    self.after_scheduler.optimizer.param_groups[0] = param_det_dict
                    self.after_scheduler.optimizer.param_groups[1] = param_freeze_det

            #---------------------------------------------------#
            #  four 联合训练
            #---------------------------------------------------#
            if joint_train_epoch > epoch_index >= 0:
                # 如果已经有单独tuning了
                if (not self.lane_first_time ) and (not self.det_first_time) and (not self.seg_first_time):
                    self.lane_first_time = True
                    self.det_first_time = True
                    self.seg_first_time = True
                    print("=====================start joint training again=====================")

                    #---------------------------------------------------#
                    #  提取通用参数
                    #---------------------------------------------------#
                    momentum = self.after_scheduler.optimizer.param_groups[0]["momentum"]
                    dampening = self.after_scheduler.optimizer.param_groups[0]["dampening"]
                    weight_decay = self.after_scheduler.optimizer.param_groups[0]["weight_decay"]
                    nesterov = self.after_scheduler.optimizer.param_groups[0]["nesterov"]

                    # 参数列表
                    param_one = self.after_scheduler.optimizer.param_groups[0]["params"]
                    param_two = self.after_scheduler.optimizer.param_groups[1]["params"]
                    param_list = param_two[0:SEG_START] + param_one + param_two[SEG_START:]

                    #---------------------------------------------------#
                    #  联合训练
                    #---------------------------------------------------#
                    param_all_dict = dict()
                    param_activate_all = param_list
                    param_all_dict["params"] = param_activate_all
                    param_all_dict["lr"] = self.lr_tuning_init
                    param_all_dict["momentum"] = momentum
                    param_all_dict["dampening"] = dampening
                    param_all_dict["weight_decay"] = weight_decay
                    param_all_dict["nesterov"] = nesterov
                    param_all_dict["initial_lr"] =  self.lr_tuning_init

                    self.after_scheduler.optimizer.param_groups.pop()
                    self.after_scheduler.optimizer.param_groups[0] = param_all_dict

            self.after_scheduler.step(epoch)
            return

        self.current_iters = epoch
        warmup_lr = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
            param_group['lr'] = lr

        if epoch >= self.warmup_iters:
            self.warmup_finished = True
            self.after_scheduler = LrScheduler(self.after_scheduler_config)(self.optimizer)
            self.by_epoch = self.after_scheduler.by_epoch
