# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Head of CurveLaneNas."""
import torch
import torch.nn as nn
from vega.modules.module import Module
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class AutoLaneHead(Module):
    """CurveLaneHead."""

    def __init__(self, desc):
        """Construct head."""
        super(AutoLaneHead, self).__init__()
        base_channel = desc["base_channel"]
        num_classes = desc["num_classes"]
        self.num_classes = num_classes

        # feature dimension
        self.stride = desc["input_size"]["anchor_stride"]
        self.input_width = desc["input_size"]["width"]
        self.input_height = desc["input_size"]["height"]
        self.interval = desc["input_size"]["interval"]
        self.feat_width = int(self.input_width / self.stride)
        self.feat_height = int(self.input_height / self.stride)
        self.points_per_line = int(self.input_height / self.interval)

        self.lane_up_pts_num = self.points_per_line + 1
        self.lane_down_pts_num = self.points_per_line + 1

        # 车道线存在与否分支
        self.conv_cls_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, num_classes, kernel_size=1, stride=1)
        )

        #---------------------------------------------------#
        #  车道线类别分支
        #---------------------------------------------------#
        self.do_lane_classification = desc["do_classify"]
        self.lane_class_name_list = desc["lane_class_list"]
        self.lane_class_num = len(self.lane_class_name_list)
        if self.do_lane_classification:
            self.conv_lane_type_head = nn.Sequential(
                nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(base_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channel, self.lane_class_num, kernel_size=1, stride=1),
            )

        #---------------------------------------------------#
        #  竖直方向的检测
        #---------------------------------------------------#
        self.conv_up_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, self.lane_up_pts_num, kernel_size=1, stride=1)
        )

        self.conv_down_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, self.lane_down_pts_num, kernel_size=1, stride=1)
        )

        # for index, m in enumerate(self.modules()):
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, input, **kwargs):
        """Forward method of this head."""
        predict_cls = self.conv_cls_conv(input).permute((0, 2, 3, 1)).contiguous()
        predict_cls = predict_cls.view(predict_cls.shape[0], -1, self.num_classes)

        predict_up = self.conv_up_conv(input).permute((0, 2, 3, 1))
        predict_down = self.conv_down_conv(input).permute((0, 2, 3, 1))
        predict_loc = torch.cat([predict_down, predict_up], -1).contiguous()
        predict_loc = predict_loc.view(predict_loc.shape[0], -1, self.lane_up_pts_num + self.lane_down_pts_num)

        if self.do_lane_classification:
            predict_lane_type = self.conv_lane_type_head(input).permute((0, 2, 3, 1)).contiguous()
            predict_lane_type = predict_lane_type.view(predict_lane_type.shape[0], -1, self.lane_class_num)

            result = dict(
                predict_cls=predict_cls,
                predict_loc=predict_loc,
                predict_lane_type=predict_lane_type
            )

        else:
            result = dict(
                predict_cls=predict_cls,
                predict_loc=predict_loc
            )

        return result

    @property
    def input_shape(self):
        """Output of backbone."""
        return self.feat_height, self.feat_width
