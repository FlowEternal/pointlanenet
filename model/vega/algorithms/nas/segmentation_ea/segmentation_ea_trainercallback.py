# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for SegmentationEA."""
import logging
import torch
from vega.common import ClassFactory, ClassType
from vega.metrics import calc_model_flops_params
from vega.trainer.callbacks import Callback

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class SegmentationEATrainerCallback(Callback):
    """Construct the trainer of Adelaide-EA."""

    def before_train(self, logs=None):
        """Be called before the training process."""
        self.config = self.trainer.config
        count_input = torch.FloatTensor(1, 3, 1024, 1024).cuda()
        flops_count, params_count = calc_model_flops_params(
            self.trainer.model, count_input)
        self.flops_count, self.params_count = flops_count * 1e-9, params_count * 1e-3
        logger.info("Flops: {:.2f} G, Params: {:.1f} K".format(self.flops_count, self.params_count))

    def after_epoch(self, epoch, logs=None):
        """Update flops and params."""
        summary_perfs = logs.get('summary_perfs', {})
        summary_perfs.update({'flops': self.flops_count, 'params': self.params_count})
        logs.update({'summary_perfs': summary_perfs})
