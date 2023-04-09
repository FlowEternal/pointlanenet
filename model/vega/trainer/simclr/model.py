# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Build model for simclr."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimclrModel(nn.Module):
    """Build simclr train model.

    param init_model: the model for downstream task, which contains backbone and head
    type init_model: nn.Module
    param linear_channel: the output channels number of simclr projection head
    type linear_channel: int
    param num_class: channel number of simclr output
    type num_class: int
    """

    def __init__(self, init_model, linear_channel=512, num_class=10):
        super(SimclrModel, self).__init__()

        self.f = []
        for name, module in init_model.named_children():
            if name.startswith("backbone"):
                self.f.append(module)
        self.f.append(nn.AdaptiveAvgPool2d((1, 1)))
        # backbone
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(self.output_channel, linear_channel, bias=False),
                               nn.BatchNorm1d(linear_channel),
                               nn.ReLU(inplace=True),
                               nn.Linear(linear_channel, num_class, bias=True))

    def forward(self, x):
        """Compute the output of simclr model."""
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

    @property
    def output_channel(self):
        """Output Channel for last conv2d."""
        return [module.out_channels for name, module in self.named_modules() if isinstance(module, nn.Conv2d)][-1]
