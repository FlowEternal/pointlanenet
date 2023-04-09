# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for AutoContrast."""
import numpy as np
import random
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class AutoAugment(object):
    """Applies AutoContrast to 'img'.

    The AutoContrast operation maximizes the the image contrast, by making the darkest pixel black
    and lightest pixel white.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    _RAND_TRANSFORMS = {
        'AutoContrast': 'AutoContrast',
        'Equalize': 'Equalize',
        'Invert': 'Invert',
        'Rotate': 'Rotate',
        'Posterize': 'Posterize',
        'Solarize': 'Solarize',
        'Color': 'Color',
        'Contrast': 'Contrast',
        'Brightness': 'Brightness',
        'Sharpness': 'Sharpness',
        'ShearX': 'Shear_X',
        'ShearY': 'Shear_Y',
        'TranslateXRel': 'Translate_X',
        'TranslateYRel': 'Translate_Y',
        # 'Cutout': 'Cutout'
    }

    def __init__(self, num=2, magnitude=9, prob=0.5, magnitude_std=0.5):
        """Construct the AutoContrast class."""
        self.num = num
        self.magnitude = magnitude
        self.prob = prob
        self.magnitude_std = magnitude_std

    def __call__(self, img):
        """Call function of AutoContrast.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        transforms = []
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        for name in self._RAND_TRANSFORMS.keys():
            if ClassFactory.is_exists(ClassType.TRANSFORM, self._RAND_TRANSFORMS[name]):
                transforms.append(ClassFactory.get_cls(ClassType.TRANSFORM, self._RAND_TRANSFORMS[name]))
        ops = np.random.choice(
            transforms, self.num)
        for op in ops:
            if self.magnitude_std and self.magnitude_std > 0:
                magnitude = random.gauss(self.magnitude, self.magnitude_std)
            magnitude = min(10, max(0, magnitude))
            img = op(magnitude)(img)
        return img
