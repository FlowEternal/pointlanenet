# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Elastic sequential (depth) transformations."""
import torch.nn as nn
from .modifier import modify_attr, restore_module_attrs


def _hook_module_in(module, inputs):
    if ElasticSequential.get_sequential_state(module):
        modify_attr(module, 'forward', lambda x: x)


def _hook_module_out(module, inputs, result):
    restore_module_attrs(module)


class ElasticSequential():
    """Elastic sequential group manager."""

    _module_hooks = dict()
    _groups = list()

    @staticmethod
    def add_group(group):
        """Add a group."""
        ElasticSequential._groups.append(group)

    @staticmethod
    def remove_group(group):
        """Remove a group."""
        idx = ElasticSequential._groups.index(group)
        if not idx == -1:
            group.destroy()
            del ElasticSequential._groups[idx]

    @staticmethod
    def groups():
        """Return an iterator over groups."""
        for g in ElasticSequential._groups:
            yield g

    @staticmethod
    def num_groups():
        """Return the number of groups."""
        return len(ElasticSequential._groups)

    @staticmethod
    def enable_sequential_transform(module):
        """Enable sequential transformation on a module."""
        if module not in ElasticSequential._module_hooks:
            h_in = module.register_forward_pre_hook(_hook_module_in)
            h_out = module.register_forward_hook(_hook_module_out)
            ElasticSequential._module_hooks[module] = (h_in, h_out)

    @staticmethod
    def disable_sequential_transform(module):
        """Disable sequential transformation on a module."""
        if module in ElasticSequential._module_hooks:
            m_hooks = ElasticSequential._module_hooks.pop(module)
            for h in m_hooks:
                h.remove()
            del module._sequential_state

    @staticmethod
    def set_sequential_state(module, state):
        """Set sequential state of a module."""
        module._sequential_state = state

    @staticmethod
    def reset_sequential_state(module):
        """Reset sequential state of a module."""
        module._sequential_state = None

    @staticmethod
    def get_sequential_state(module):
        """Get sequential state of a module."""
        if not hasattr(module, '_sequential_state'):
            module._sequential_state = None
        return module._sequential_state


class ElasticSequentialGroup():
    """Module group with elastic sequential dimensions."""

    def __init__(self, *args):
        module_groups = []
        for m in args:
            if isinstance(m, nn.Module):
                group = [m]
            elif isinstance(m, (list, tuple)):
                group = list(m)
            else:
                raise ValueError('invalid args')
            module_groups.append(group)
        self.max_depth = len(module_groups)
        self.module_groups = module_groups
        self.enable_sequential_transform()
        ElasticSequential.add_group(self)

    def destroy(self):
        """Destroy group."""
        self.reset_sequential_idx()
        self.disable_sequential_transform()

    def enable_sequential_transform(self):
        """Enable sequential transformation of group modules."""
        for m in self.modules():
            ElasticSequential.enable_sequential_transform(m)

    def disable_sequential_transform(self):
        """Disable sequential transformation of group modules."""
        for m in self.modules():
            ElasticSequential.disable_sequential_transform(m)

    def set_depth_ratio(self, ratio):
        """Set group depth by ratio of the max depth."""
        if ratio is None:
            self.reset_sequential_idx()
            return
        depth = int(self.max_depth * ratio)
        self.set_depth(depth)

    def set_depth(self, depth):
        """Set group depth."""
        if depth is None:
            self.reset_sequential_idx()
            return
        if depth > self.max_depth:
            raise ValueError('depth out of range')
        self.set_sequential_idx(list(range(depth)), reverse=True)

    def set_sequential_idx(self, idx, reverse=False):
        """Set group sequential index."""
        if isinstance(idx, int):
            idx = [idx]
        for i, m_group in enumerate(self.module_groups):
            state = 1 if i in idx else 0
            state = 1 - state if reverse else state
            for m in m_group:
                ElasticSequential.set_sequential_state(m, state)

    def reset_sequential_idx(self):
        """Reset group sequential index."""
        for m in self.modules():
            ElasticSequential.reset_sequential_state(m)

    def modules(self, active=False):
        """Return an iterator over all group modules."""
        for m_group in self.module_groups:
            for m in m_group:
                if not active or not ElasticSequential.get_sequential_state(m):
                    yield m
