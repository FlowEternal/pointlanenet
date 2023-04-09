# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch Trainer."""

import os
import torch
import numpy as np
import vega
from vega.trainer.trainer_base import TrainerBase
from vega.modules.loss import Loss
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.trainer.modules.optimizer import Optimizer
from vega.common import ClassFactory, ClassType

#---------------------------------------------------#
#  车道线类别
#---------------------------------------------------#
CLASS_TYPE_LIST = ["normal_abnormal",
                   "single_double",
                   "solid_dash",
                   "white_yellow"]

#---------------------------------------------------#
#  检测相关
#---------------------------------------------------#
from vega.networks.pytorch.detectors.det_map import eval_mAP

#---------------------------------------------------#
#  分割评估相关函数
#---------------------------------------------------#
from vega.networks.pytorch.detectors.metric_seg import IntersectionOverUnion

# 语义分割类别定义
seg_class_list = {"background":0,
                  "pedestrian_area":1,
                  "self_area":2,
                  "obstacle_area":3,
                  "road_area":4,
                  "marking_area":5,
                  "vehicle_area":6,
                  "marking_general_area":7,
                  "marking_pavement_area":8
                  }

seg_class_list_key = seg_class_list.keys()

# 语义分割可视化颜色定义
seg_class_color = {"background":(0,0,0),
                   "pedestrian_area":(0,255,0),
                  "self_area":(0,0,255),
                  "obstacle_area":(255,0,0),
                  "road_area":(255,0,255),
                  "marking_area":(0,255,255),
                  "vehicle_area":(128,255,255),
                  "marking_general_area":(255,128,0),
                  "marking_pavement_area": (128,0,255)
                   }

# 语义分割可视化颜色定义
seg_class_color_id = {0:(0,0,0),
                      1:(0,255,0),
                      2:(0,0,255),
                      3:(255,0,0),
                      4:(255,0,255),
                      5:(0,255,255),
                      6:(128,255,255),
                      7:(255,128,0),
                      8:(128,0,255)
                   }


@ClassFactory.register(ClassType.TRAINER)
class TrainerTorch(TrainerBase):
    """Trainer torch class."""

    def build(self):
        """Build the trainer by assembling the necessary components."""
        super().build()
        if self.optimizer is None:
            self.optimizer = Optimizer()(model=self.model, distributed=self.distributed)
        if hasattr(self.model, 'add_loss'):
            loss_cls = Loss()()
            self.model.add_loss(loss_cls)
            self.loss = self.model.overall_loss()
        else:
            self.loss = Loss()()
        if self.config.adaptive_muti_loss and hasattr(self.loss, "adaptive_muti_loss"):
            self.loss.adaptive_muti_loss(save_path=self.get_local_worker_path(self.step_name, self.worker_id),
                                         weight=self.config.loss_weight)
        if self.lr_scheduler is None:
            self.lr_scheduler = LrScheduler()(self.optimizer)
        if self.actions_list is not None:
            self.total_optimizer = self.optimizer
            self.total_loss = self.loss
            self.total_lr_scheduler = self.lr_scheduler
        # Some trainer has different train batch size from valid batch
        self.train_metrics = self._init_metrics()
        self.valid_metrics = self._init_metrics()
        self._init_horovod_setting()
        if self.use_amp:
            from apex import amp
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level='O1')

    def _set_default_funcs(self):
        self.make_batch = self._default_make_batch
        if isinstance(self.config.optimizer, list):
            self.train_step = self._multi_train_step
        else:
            self.train_step = self._default_train_step
        self.valid_step = self._default_valid_step

    def _set_condition(self):
        self._init_distributed_setting()
        torch.manual_seed(self.config.seed)
        self._init_setting()

    def _init_setting(self):
        """Init CUDA setting."""
        if vega.is_gpu_device():
            import torch.cuda
            self.config.device = vega.is_gpu_device() if vega.is_gpu_device() is not True else 0
            if self.distributed:
                torch.cuda.set_device(self._local_rank_id)
            torch.cuda.manual_seed(self.config.seed)
        elif vega.is_npu_device():
            import torch.npu
            device = "npu:{}".format(os.environ.get('DEVICE_ID', 0))
            torch.npu.set_device(device)
            torch.npu.manual_seed(self.config.seed)
        elif vega.is_cpu_device():
            self.config.device = -1
            return
        else:
            raise ValueError('Set a correct device: cuda or npu.')

    def _init_distributed_setting(self):
        if self.distributed:
            import horovod.torch as hvd
            self._world_size = hvd.size()
            self._rank_id = hvd.rank()
            self._local_rank_id = hvd.local_rank()

    def _init_horovod_setting(self):
        """Init horovod setting."""
        self.is_chief = True
        if self.distributed:
            import horovod.torch as hvd
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
            if hvd.rank() != 0:
                self.is_chief = False
            else:
                self.is_chief = True

    def _train_epoch(self):
        self.model.train()

        # if self.config.optimizer.adjust_lane_type:
        #     def fix_bn(m):
        #         classname = m.__class__.__name__
        #         if classname.find('BatchNorm') != -1:
        #             m.eval()
        #     self.model.apply(fix_bn)

        for batch_index, batch in enumerate(self.train_loader):
            if self.config.max_train_steps and batch_index > self.config.max_train_steps:
                return
            batch = self.make_batch(batch)
            batch_logs = {'train_batch': batch}
            self.callbacks.before_train_step(batch_index, batch_logs)
            train_batch_output = self.train_step(batch)
            batch_logs.update(train_batch_output)
            if self.config.is_detection_trainer:
                batch_logs.update({'is_detection_trainer': True})
            self.callbacks.after_train_step(batch_index, batch_logs)

    def _valid_epoch(self):
        self.callbacks.before_valid()
        valid_logs = None

        TRAIN_SEG = False
        metric_evaluator_iou = IntersectionOverUnion(n_classes=len(seg_class_list))

        TRAIN_DET = False
        root_dir = ""

        TRAIN_LANE_TYPE = False
        metric_list_all = list()

        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.valid_loader):
                batch = self.make_batch(batch)

                # 判断是否进行分割
                if "gt_seg" in batch[1]:
                    TRAIN_SEG = True

                if "gt_det" in batch[1]:
                    TRAIN_DET = True
                    root_dir = os.path.dirname(os.path.dirname(batch[1]["annotation_path"][0]))

                batch_logs = {'valid_batch': batch}
                self.callbacks.before_valid_step(batch_index, batch_logs)

                if TRAIN_SEG:
                    valid_batch_output = self.valid_step(batch,iou_metric=metric_evaluator_iou)
                else:
                    valid_batch_output = self.valid_step(batch)

                if self.config.train_lane:
                    self.callbacks.after_valid_step(batch_index, valid_batch_output)

        # 车道线打印
        if self.config.train_lane:
            print("=================== 车道线指标 ===================")
            self.callbacks.after_valid(valid_logs)

        # 目标检测mAP计算
        if TRAIN_DET:
            import warnings
            warnings.filterwarnings("ignore")
            gt_path = os.path.join(root_dir,"labels_object_tmp")
            pred_path = os.path.join(root_dir,"labels_object_tmp_pred")
            mAP = eval_mAP(root_dir,valid_gt_path=gt_path,valid_pred_path= pred_path,use_07_metric=False)
            # display result
            print("=================== 目标检测指标 ===================")
            print("mAP %.4f" % mAP)
            print()

        # 语义分割mIOU计算
        if TRAIN_SEG:
            print("================== 语义分割指标 ==================")
            metric_info = ""
            scores = metric_evaluator_iou.compute()
            for key, value in zip(seg_class_list_key, scores):
                metric_info += key + " %.3f" % value + " "
            print(metric_info)


    def _default_make_batch(self, batch):
        """Unpack batch to get input and target."""
        if not vega.is_cpu_device():
            batch = self._set_device(batch)
        return batch

    def _set_device(self, data):
        if torch.is_tensor(data):
            if vega.is_gpu_device():
                return data.cuda()
            else:
                return data.npu()
        if isinstance(data, dict):
            return {k: self._set_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._set_device(v) for v in data]
        elif isinstance(data, tuple):
            return tuple([self._set_device(v) for v in data])
        return data

    def _default_train_step(self, batch):
        self.optimizer.zero_grad()
        input, target = None, None
        if isinstance(batch, dict):
            output = self.model(**batch)
        elif isinstance(batch, list) and isinstance(batch[0], dict):
            output = self.model(batch)
        else:
            # classification
            input, target = batch
            if self.config.mixup:
                mixup_ratio = np.random.beta(0.1, 0.1)
                mixed_x, y_a, y_b = self._mixup_batch(input, target, mixup_ratio)
                output = self.model(mixed_x)
            else:
                output = self.model(input) if not isinstance(input, dict) else self.model(**input)
        # loss
        if self.config.mixup:
            loss = self._mixup_loss(self.loss, output, y_a, y_b, mixup_ratio)
        else:
            loss = self.loss(output, target)
        if self.use_amp:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
                self.optimizer.synchronize()
            with self.optimizer.skip_synchronize():
                self.optimizer.step()
        else:
            loss.backward()
            if self.config.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return {'loss': loss.item(),
                'train_batch_output': output,
                'lr': self.lr_scheduler.get_lr()}

    def _multi_train_step(self, batch):
        train_batch_output = None
        for opt_name, sub_opt in self.optimizer.get_opts():
            self.optimizer = sub_opt.get('opt')
            self.loss = sub_opt.get('loss')
            self.lr_scheduler = sub_opt.get('lr')
            train_batch_output = self._default_train_step(batch)
        return train_batch_output

    def _default_valid_step(self, batch):
        if isinstance(batch, dict):
            output = self.model(**batch)
        elif isinstance(batch, list) and isinstance(batch[0], dict):
            output = self.model(batch)
        else:
            input, target = batch
            output = self.model(input) if not isinstance(input, dict) else self.model(**input)
        return {'valid_batch_output': output}

    def _mixup_batch(self, x, y, ratio):
        indices = torch.randperm(x.shape[0])
        mixed_x = ratio * x + (1 - ratio) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b

    def _mixup_loss(self, loss, pred, y_a, y_b, ratio):
        return ratio * loss(pred, y_a) + (1 - ratio) * loss(pred, y_b)
