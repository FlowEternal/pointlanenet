# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for AutoContrast."""
from PIL import ImageOps
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class AutoContrast(object):
    """Applies AutoContrast to 'img'.

    The AutoContrast operation maximizes the the image contrast, by making the darkest pixel black
    and lightest pixel white.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the AutoContrast class."""
        self.level = level

    def __call__(self, img):
        """Call function of AutoContrast.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        return ImageOps.autocontrast(img.convert('RGB'))
