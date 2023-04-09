# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is Embeddings classes."""
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class BertEmbeddings(Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = ops.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = ops.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = ops.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = ops.Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids=None):
        """Get embeddings."""
        seq_length = input_ids.size(1)
        position_ids = ops.arange(seq_length, dtype='long', device=input_ids.device)
        position_ids = ops.expand_as(ops.unsqueeze(position_ids, 0), input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
