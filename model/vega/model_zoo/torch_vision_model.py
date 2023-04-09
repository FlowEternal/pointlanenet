# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torchvision networks and models."""
from types import ModuleType
from torchvision import models as torchvision_models
from vega.common import ClassType, ClassFactory


def import_all_torchvision_models():
    """Import all torchvision networks and models."""

    def _register_models_from_current_module_scope(module):
        for _name in dir(module):
            if _name.startswith("_"):
                continue
            _cls = getattr(module, _name)
            if isinstance(_cls, ModuleType):
                continue
            if ClassFactory.is_exists(ClassType.NETWORK, 'torchvision_' + _cls.__name__):
                continue
            ClassFactory.register_cls(_cls, ClassType.NETWORK, alias='torchvision_' + _cls.__name__)

    _register_models_from_current_module_scope(torchvision_models)
    _register_models_from_current_module_scope(torchvision_models.segmentation)
    _register_models_from_current_module_scope(torchvision_models.detection)
    _register_models_from_current_module_scope(torchvision_models.video)
