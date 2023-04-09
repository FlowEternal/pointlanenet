# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ConnectionsArchParamsCombiner."""
from collections import deque
from vega.modules.operators import ops
from vega.modules.connections import Add
from vega.modules.connections import Sequential
import vega


class ConnectionsArchParamsCombiner(object):
    """Get ConnectionsArchParamsCombiner."""

    def __init__(self):
        self.pre_conv = None
        self.arch_type = None
        self.search_space = []
        self.conditions = deque()
        self.forbidden = []

    def get_search_space_by_arch_type(self, module, arch_type):
        """Get Search Space."""
        self.arch_type = arch_type
        self._traversal(module)
        return self.search_space

    def combine(self, module):
        """Decode modules arch params."""
        self._traversal(module)
        module.set_arch_params({k: v for k, v in module._arch_params.items() if k not in self.forbidden})
        if module._arch_params_type == 'Prune':
            for k, v in self.conditions:
                if module._arch_params.get(v):
                    module._arch_params[k] = module._arch_params.get(v)
        return self.search_space

    def _traversal(self, module):
        """Traversal search space and conditions."""
        if isinstance(module, Add):
            self._traversal_add_connections(module)
        elif isinstance(module, ops.Conv2d):
            if self.pre_conv:
                self.add_condition(module.name + '.in_channels', self.pre_conv.name + '.out_channels')
            self.pre_conv = module
        elif isinstance(module, ops.BatchNorm2d):
            self.add_condition(module.name + '.num_features', self.pre_conv.name + '.out_channels')
        elif isinstance(module, ops.Linear):
            self.add_condition(module.name + '.in_features', self.pre_conv.name + '.out_channels')
        elif isinstance(module, Sequential):
            for child in module.children():
                self._traversal(child)

    def _traversal_add_connections(self, module):
        last_convs = []
        last_bns = []
        for child in module.children():
            if isinstance(child, ops.Conv2d):
                add_convs = [child]
            elif isinstance(child, ops.Identity):
                continue
            else:
                add_convs = [conv for name, conv in child.named_modules() if isinstance(conv, ops.Conv2d)]
                add_bns = [bn for name, bn in child.named_modules() if isinstance(bn, ops.BatchNorm2d)]
            if add_convs:
                last_convs.append(add_convs[-1])
                if vega.is_ms_backend():
                    last_bns.append(add_bns[-1])
        tmp_pre_conv = self.pre_conv
        for child in module.children():
            self.pre_conv = tmp_pre_conv
            self._traversal(child)
        if len(last_convs) > 1:
            self.pre_conv = last_convs[0]
            last_convs = last_convs[1:]
        else:
            self.pre_conv = tmp_pre_conv
        for conv in last_convs:
            self.add_condition(conv.name + '.out_channels', self.pre_conv.name + '.out_channels')
        # The out_channels value of the jump node is the same as that of the previous nodes
        # remove from the search space.
        if len(last_convs) == 1:
            self.add_forbidden(last_convs[0].name + '.out_channels')
            self.add_condition(last_convs[0].name + '.out_channels', self.pre_conv.name + '.out_channels')
            if vega.is_ms_backend():
                self.add_condition(last_bns[-1].name + '.num_features', self.pre_conv.name + '.out_channels')
        else:
            for last_conv in last_convs:
                if self.pre_conv == last_conv:
                    continue
                self.add_forbidden(last_conv.name + '.out_channels')
                self.add_condition(last_convs[0].name + '.out_channels', self.pre_conv.name + '.out_channels')
                for k, v in [(k, v) for k, v in self.conditions if v == last_conv.name + '.out_channels']:
                    self.add_condition(k, self.pre_conv.name + '.out_channels')
        self.pre_conv = last_convs[0]

    def add_condition(self, name, value):
        """Add condition."""
        self.conditions.append((name, value))

    def add_forbidden(self, name):
        """Add condition."""
        self.forbidden.append(name)
