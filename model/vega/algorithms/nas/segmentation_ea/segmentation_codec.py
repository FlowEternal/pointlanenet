# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Codec of Prune EA."""
import re
from vega.common import ClassFactory, ClassType
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class SegmentationCodec(Codec):
    """Class of Prune EA Codec.

    :param codec_name: name of this codec
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space, **kwargs):
        """Init PruneCodec."""
        super(SegmentationCodec, self).__init__(search_space, **kwargs)

    def decode(self, arch_string):
        """Decode the code to Network Desc.

        :param code: input code
        :type code: list of int
        :return: network desc
        :rtype: NetworkDesc
        """
        print('Begin decoding architecture ' + str(arch_string))
        model_type = arch_string[0]
        if model_type == 'r':
            # ResNet
            # e.g. r101_48_1121-21-11111111111-21111111111
            m = re.match(r'r(.*)_(.*)_(.*)', arch_string)
            base_depth, base_channel = map(int, m.groups()[:-1])
            arch = m.groups()[-1]
            return dict(base_depth=base_depth, groups=1, base_width=base_channel, base_channel=base_channel,
                        arch=arch)
        elif model_type == 'x':
            # e.g. x101(24x4d)_48_1121-21-11111111111-21111111111
            m = re.match(r'x(\d+)\D*(\d+)x(\d+)d.*_(\d+)_(.*)',
                         arch_string)
            base_depth, groups, base_width, base_channel = map(int, m.groups()[:-1])
            arch = m.groups()[-1]
            return dict(base_depth=base_depth, groups=groups, base_width=base_width, base_channel=base_channel,
                        arch=arch)
        if model_type == 'b':
            # ResNet
            # e.g. r101_48_1121-21-11111111111-21111111111
            m = re.match(r'(.*)_(.*)_(.*)', arch_string)
            base_depth, base_channel, arch = m.groups()
            base_channel = int(base_channel)
            return dict(base_depth=base_depth, groups=1, base_width=base_channel, base_channel=base_channel,
                        arch=arch)
        else:
            raise ValueError('Cannot parse arch code {}'.format(arch_string))
