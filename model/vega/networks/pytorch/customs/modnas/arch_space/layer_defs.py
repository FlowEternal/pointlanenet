# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Dataflow defining components in Layers."""
import itertools
import torch
from modnas.registry.layer_def import register


class MergerBase():
    """Base layer dataflow merger class."""

    def chn_out(self, chn_states):
        """Return number of channels in merged output."""
        raise NotImplementedError

    def merge(self, states):
        """Return merged output from inputs."""
        raise NotImplementedError

    def merge_range(self, num_states):
        """Return indices of merged inputs."""
        raise NotImplementedError


@register
class ConcatMerger(MergerBase):
    """Merger that outputs concatenation of inputs."""

    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        """Return number of channels in merged output."""
        return sum(chn_states[self.start:])

    def merge(self, states):
        """Return merged output from inputs."""
        return torch.cat(states[self.start:], dim=1)

    def merge_range(self, num_states):
        """Return indices of merged inputs."""
        return range(self.start, num_states)


@register
class AvgMerger(MergerBase):
    """Merger that outputs mean of inputs."""

    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        """Return number of channels in merged output."""
        return chn_states[-1]

    def merge(self, states):
        """Return merged output from inputs."""
        return sum(states[self.start:]) / len(states)

    def merge_range(self, num_states):
        """Return indices of merged inputs."""
        return range(self.start, num_states)


@register
class SumMerger(MergerBase):
    """Merger that outputs sum of inputs."""

    def __init__(self, start=0):
        super().__init__()
        self.start = start

    def chn_out(self, chn_states):
        """Return number of channels in merged output."""
        return chn_states[-1]

    def merge(self, states):
        """Return merged output from inputs."""
        return sum(states[self.start:])

    def merge_range(self, num_states):
        """Return indices of merged inputs."""
        return range(self.start, num_states)


@register
class LastMerger(MergerBase):
    """Merger that outputs the last one of inputs."""

    def __init__(self, start=0):
        del start
        super().__init__()

    def chn_out(self, chn_states):
        """Return number of channels in merged output."""
        return chn_states[-1]

    def merge(self, states):
        """Return merged output from inputs."""
        return states[-1]

    def merge_range(self, num_states):
        """Return indices of merged inputs."""
        return (num_states - 1, )


class EnumeratorBase():
    """Base layer dataflow input enumerator class."""

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        raise NotImplementedError

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        raise NotImplementedError


@register
class CombinationEnumerator(EnumeratorBase):
    """Enumerator that enums all combinations of inputs."""

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        return itertools.combinations(range(n_states), n_inputs)

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        return len(list(itertools.combinations(range(n_states), n_inputs)))


@register
class LastNEnumerator(EnumeratorBase):
    """Enumerator that enums last N of inputs."""

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        return ([n_states - i - 1 for i in range(n_inputs)], )

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        return 1


@register
class FirstNEnumerator(EnumeratorBase):
    """Enumerator that enums first N of inputs."""

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        return ([i for i in range(n_inputs)], )

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        return 1


@register
class TreeEnumerator(EnumeratorBase):
    """Enumerator that enums inputs as trees."""

    def __init__(self, width=2):
        super().__init__()
        self.width = width

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        return ([(n_states - 1 + i) // self.width for i in range(n_inputs)], )

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        return 1


@register
class ReverseEnumerator(EnumeratorBase):
    """Enumerator that enums inputs in reverse order."""

    def enum(self, n_states, n_inputs):
        """Return enumerated indices from all inputs."""
        return ([n_states - i - 1 for i in range(n_states)], )

    def len_enum(self, n_states, n_inputs):
        """Return number of enumerated inputs."""
        return 1


class AllocatorBase():
    """Base layer dataflow input allocator class."""

    def __init__(self, n_inputs, n_states):
        self.n_inputs = n_inputs
        self.n_states = n_states

    def alloc(self, states, sidx, cur_state):
        """Return allocated input from previous states."""
        raise NotImplementedError

    def chn_in(self, chn_states, sidx, cur_state):
        """Return number of channels of allocated input."""
        raise NotImplementedError


@register
class EvenSplitAllocator(AllocatorBase):
    """Allocator that splits channels for each input."""

    def __init__(self, n_inputs, n_states):
        super().__init__(n_inputs, n_states)
        self.tot_states = n_inputs + n_states
        self.slice_map = {}

    def alloc(self, states, sidx, cur_state):
        """Return allocated input from previous states."""
        ret = []
        for s, si in zip(states, sidx):
            s_slice = self.slice_map[(si, cur_state)]
            s_in = s[:, s_slice]
            ret.append(s_in)
        return ret

    def chn_in(self, chn_states, sidx, cur_state):
        """Return number of channels of allocated input."""
        chn_list = []
        for (chn_s, si) in zip(chn_states, sidx):
            etot = min(self.n_states, self.tot_states - si - 1)
            eidx = cur_state - max(self.n_inputs, si + 1)
            c_in = chn_s - (chn_s // etot) * eidx if eidx == etot - 1 else chn_s // etot
            chn = chn_s // etot
            end = chn_s if eidx == etot - 1 else chn * (eidx + 1)
            s_slice = slice(chn * eidx, end)
            self.slice_map[(si, cur_state)] = s_slice
            chn_list.append(c_in)
        return chn_list


@register
class ReplicateAllocator(AllocatorBase):
    """Allocator that replicate states for each input."""

    def alloc(self, states, sidx, cur_state):
        """Return allocated input from previous states."""
        return states

    def chn_in(self, chn_states, sidx, cur_state):
        """Return number of channels of allocated input."""
        return chn_states
