# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""cross Entropy Weight Decay Loss."""

import tensorflow.compat.v1 as tf
import vega
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.LOSS)
class CrossEntropyLoss(object):
    """Loss lass of Cross Entroppy Weight Decay.

    :param cross_entropy: cross entropy function name
    :type cross_entropy: str
    :param weight_decay: weight decay for regulation
    :type weight_decay: float
    """

    def __init__(self, ignore_index=None):
        self.cross_entropy = tf.losses.sparse_softmax_cross_entropy
        self.ignore_index = ignore_index

    def __call__(self, logits, labels):
        """Loss forward function."""
        weights = 1.0
        if self.ignore_index is not None:
            logits, labels, weights = self.exclude_ignore_index(logits, labels)
        cross_entropy_loss = self.cross_entropy(logits=logits, labels=labels, weights=weights)
        return cross_entropy_loss

    def exclude_ignore_index(self, logits, labels):
        """Ignore certain index."""
        logits = tf.transpose(logits, [0, 2, 3, 1])
        if vega.is_gpu_device():
            indices = tf.where(tf.not_equal(labels, self.ignore_index))
            labels = tf.cast(tf.gather_nd(labels, indices), tf.int32)
            logits = tf.gather_nd(logits, indices)
            return logits, labels, 1.0
        elif vega.is_npu_device():
            weights = tf.not_equal(labels, self.ignore_index)
            labels = tf.multiply(labels, tf.cast(weights, labels.dtype))
            return logits, labels, tf.to_float(weights)
