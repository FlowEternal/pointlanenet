# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is Encode classes."""
import copy
from vega.modules.module import Module
from vega.modules.operators import ops
from vega.common.class_factory import ClassType, ClassFactory
from vega.modules.connections import ModuleList
from .attention import BertAttention
from .output import BertOutput


@ClassFactory.register(ClassType.NETWORK)
class BertEncoder(Module):
    """Bert Encoder."""

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layer.append(copy.deepcopy(layer))

    def call(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """Encode bert layers."""
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertLayer(Module):
    """Bert Layer."""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def call(self, hidden_states, attention_mask):
        """Call bert layer."""
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertIntermediate(Module):
    """Bert Intermediate."""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = ops.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            active_fn = {"gelu": ops.gelu, "relu": ops.relu, "swish": ops.swish}
            self.intermediate_act_fn = active_fn[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states):
        """Call BertIntermediate."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
