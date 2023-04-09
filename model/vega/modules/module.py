# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Module Define."""

from copy import deepcopy
from collections import OrderedDict
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops
from vega.modules.operators.functions.serializable import ModuleSerializable


@ClassFactory.register(ClassType.NETWORK)
class Module(ModuleSerializable, ops.Module):
    """Module."""

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__()
        self._losses = OrderedDict()

    def set_module(self, names, layer):
        """Set Models by name."""
        parent_model = self
        if not isinstance(names, list):
            names_path = names.split('.')
        else:
            names_path = deepcopy(names)
        next_names = names_path.pop(0)
        if not names_path:
            self.add_module(names[0], layer)
        else:
            next_model = getattr(parent_model, next_names)
            next_model.set_module(names_path, layer)

    @property
    def out_channels(self):
        """Output Channel for Module."""
        convs = [conv.out_channels for name, conv in self.named_modules() if conv.__class__.__name__ == 'Conv2d']
        return convs[-1] if convs else None

    @out_channels.setter
    def out_channels(self, value):
        """Output Channel for Module."""
        if not value:
            return
        convs = [conv for name, conv in self.named_modules() if conv.__class__.__name__ == 'Conv2d']
        if convs:
            convs[-1].out_channels = value

    def add_loss(self, loss):
        """Add a loss function into module."""
        self._losses[loss.__class__.__name__] = loss

    @property
    def loss(self):
        """Define loss name or class."""
        return None

    def _create_loss(self):
        """Create loss class."""
        if self.loss is None:
            return
        if isinstance(self.loss, str):
            loss_cls = ClassFactory.get_cls(ClassType.LOSS, self.loss)
            desc = self.desc.get('loss')
            loss_obj = loss_cls(**desc) if desc is not None else loss_cls()
        else:
            loss_obj = self.loss
        self.add_loss(loss_obj)

    def overall_loss(self):
        """Call loss function, default sum all losses."""
        self._create_loss()
        from vega.modules.loss.multiloss import MultiLoss
        return MultiLoss(*list(self._losses.values()))
