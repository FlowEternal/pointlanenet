# ==================================================================
# Author    : Dongxu Zhan
# Time      : 2021/7/25 13:00
# File      : mtl_detection.py
# Function  : detection part
# ==================================================================

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F

from vega.networks.pytorch.detectors.det.fpn import *
from vega.networks.pytorch.detectors.det.heads import *
from vega.networks.pytorch.detectors.det.anchors import *
import numpy as np

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out



class DetDecoder(nn.Module):
    def __init__(self,
                 fpn_in_channels,
                 class_num = 5,
                 ):
        super(DetDecoder, self).__init__()
        self.class_num = class_num
        self.fpn_in_channels = fpn_in_channels[-3:]

        # TODO
        self.anchor_generator = Anchors(
            ratios = np.array([0.5,1,2]),
        )

        self.num_anchors = self.anchor_generator.num_anchors

        # detection header
        self.fpn = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256),
            use_asff = False
        )

        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_classes=self.class_num
        )

        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5   # xywha
        )


    def forward(self, input_features, guide_features=None):
        det_header_input = input_features[-3:]
        features = self.fpn(det_header_input)
        cls_score = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(feature) for feature in features], dim=1)

        return cls_score, bbox_pred





