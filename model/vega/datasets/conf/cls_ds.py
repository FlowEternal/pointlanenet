# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig
from vega.common import ConfigSerializable


class ClassificationDatasetCommonConfig(BaseConfig):
    """Default Optim Config."""

    data_path = None
    batch_size = 1
    shuffle = False
    drop_last = True
    n_class = None
    train_portion = 1.0
    n_images = None
    cached = True
    transforms = []
    num_workers = 1
    distributed = False
    pin_memory = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ClassificationDatasetCommon = {"data_path": {"type": str},
                                             "batch_size": {"type": int},
                                             "shuffle": {"type": bool},
                                             "drop_last": {"type": bool},
                                             "n_class": {"type": int},
                                             "train_portion": {"type": (int, float)},
                                             "n_images": {"type": (int, None)},
                                             "cached": {"type": bool},
                                             "transforms": {"type": list},
                                             "num_workers": {"type": int},
                                             "distributed": {"type": bool},
                                             "pin_memory": {"type": bool}
                                             }
        return rules_ClassificationDatasetCommon


class ClassificationDatasetTraineConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = True
    transforms = [
        dict(type='Resize', size=[256, 256]),
        dict(type='RandomCrop', size=[224, 224]),
        dict(type='RandomHorizontalFlip'),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetValConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = False
    transforms = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetTestConfig(ClassificationDatasetCommonConfig):
    """Default Cifar10 config."""

    shuffle = False
    transforms = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=[0.50, 0.5, 0.5], std=[0.50, 0.5, 0.5])]


class ClassificationDatasetConfig(ConfigSerializable):
    """Default Dataset config for Cifar10."""

    common = ClassificationDatasetCommonConfig
    train = ClassificationDatasetTraineConfig
    val = ClassificationDatasetValConfig
    test = ClassificationDatasetTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_ClassificationDataset = {"common": {"type": dict},
                                       "train": {"type": dict},
                                       "val": {"type": dict},
                                       "test": {"type": dict}
                                       }
        return rules_ClassificationDataset

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
