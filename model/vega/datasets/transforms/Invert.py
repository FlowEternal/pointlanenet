# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Invert."""
from PIL import ImageOps
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Invert(object):
    """Applies Invert to 'img'.

    The Invert operation inverts the pixels of the image.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Invert class."""
        self.level = level

    def __call__(self, img):
        """Call function of Invert.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        return ImageOps.invert(img.convert('RGB'))
