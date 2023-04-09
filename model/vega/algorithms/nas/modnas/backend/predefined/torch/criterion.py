# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Implementation of Criterions (Loss functions)."""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from modnas.registry.construct import build as build_constructor
from modnas.registry.criterion import register, build


def get_criterion(config, device_ids=None):
    """Return a new Criterion."""
    n_parallel = 1 if device_ids is None else len(device_ids)
    criterion = build(config)
    if n_parallel > 1 and isinstance(criterion, torch.nn.Module):
        criterion = torch.nn.DataParallel(criterion, device_ids=device_ids).module
    return criterion


def torch_criterion_wrapper(cls):
    """Return a Criterion class that wraps a torch loss function."""

    def call_fn(self, loss, estim, *args):
        y_pred, y_true = args[0], args[-1]
        return cls.__call__(self, y_pred, y_true)

    new_cls = type(cls.__name__, (cls, ), {'__call__': call_fn})
    return new_cls


def label_smoothing(y_pred, y_true, eta):
    """Return label smoothed target."""
    n_classes = y_pred.size(1)
    # convert to one-hot
    y_true = torch.unsqueeze(y_true, 1)
    soft_y_true = torch.zeros_like(y_pred)
    soft_y_true.scatter_(1, y_true.to(dtype=torch.long), 1)
    # label smoothing
    soft_y_true = soft_y_true * (1 - eta) + eta / n_classes * 1
    return soft_y_true


def cross_entropy_soft_target(y_pred, target):
    """Return soft target cross entropy loss."""
    return torch.mean(torch.sum(-target * F.log_softmax(y_pred, dim=-1), 1))


class CrossEntropyLabelSmoothingLoss(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta

    def forward(self, y_pred, y_true):
        """Return loss."""
        soft_y_true = label_smoothing(y_pred, y_true, self.eta)
        return cross_entropy_soft_target(y_pred, soft_y_true)


@register
class MixUpLoss():
    """Apply MIXUP loss."""

    def __init__(self, crit_conf, alpha=0.3, use_flip=True):
        self.alpha = alpha
        self.use_flip = use_flip
        self.criterion = build(crit_conf)

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        alpha = self.alpha
        lam = random.betavariate(alpha, alpha) if alpha > 0 else 1
        if self.use_flip:
            alt_X = torch.flip(X, dims=[0])
            alt_y_true = torch.flip(y_true, dims=[0])
        else:
            index = list(range(X.size(0)))
            random.shuffle(index)
            alt_X = X[index, :]
            alt_y_true = y_true[index, :]
        mixed_x = lam * X + (1 - lam) * alt_X
        mixed_y_pred = estim.model_output(mixed_x)
        loss = loss or 0
        return lam * self.criterion(loss, estim, mixed_y_pred, mixed_x, y_true) + (1 - lam) * self.criterion(
            loss, estim, mixed_y_pred, mixed_x, alt_y_true)


@register
class AuxiliaryLoss():
    """Apply Auxiliary loss."""

    def __init__(self, aux_ratio=0.4, loss_type='ce', forward_func='forward_aux'):
        super().__init__()
        self.aux_ratio = aux_ratio
        self.fwd_func = forward_func
        if loss_type == 'ce':
            self.loss_func = F.cross_entropy
        else:
            raise ValueError('unsupported loss type: {}'.format(loss_type))

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        aux_logits = estim.model_output(X, attr=self.fwd_func)
        if aux_logits is None:
            return loss
        aux_loss = self.loss_func(aux_logits, y_true).to(device=X.device)
        return loss + self.aux_ratio * aux_loss


@register
class KnowledgeDistillLoss():
    """Apply Knowledge Distillation."""

    def __init__(self, kd_model_constructor=None, kd_model=None, kd_ratio=0.5, loss_scale=1., loss_type='ce'):
        super().__init__()
        self.kd_model_constructor = kd_model_constructor
        self.kd_model = kd_model
        self.kd_ratio = kd_ratio
        self.loss_scale = loss_scale
        if loss_type == 'ce':
            self.loss_func = lambda y_pred, target: cross_entropy_soft_target(y_pred, F.softmax(target, dim=-1))
        elif loss_type == 'mse':
            self.loss_func = F.mse_loss
        else:
            raise ValueError('unsupported loss_type: {}'.format(loss_type))

    def _load_model(self, kd_model, kd_model_constructor):
        if not isinstance(kd_model_constructor, list):
            kd_model_constructor = [kd_model_constructor]
        for con_conf in kd_model_constructor:
            kd_model = build_constructor(con_conf)(kd_model)
        return kd_model

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        if self.kd_model is None and self.kd_model_constructor is not None:
            self.kd_model = self._load_model(self.kd_model, self.kd_model_constructor)
        with torch.no_grad():
            self.kd_model.to(device=X.device)
            soft_logits = self.kd_model(X)
        kd_loss = self.loss_func(y_pred, soft_logits).to(device=loss.device)
        loss = self.loss_scale * ((1 - self.kd_ratio) * loss + self.kd_ratio * kd_loss)
        return loss


class AggMetricsLoss():
    """Compute loss from Metrics."""

    def __init__(self, metrics, target_val=None, target_decay=0.1):
        super().__init__()
        if target_val is not None:
            target_val = float(target_val)
        self.target_val = target_val
        self.target_decay = target_decay
        self.metrics = metrics

    def _get_metrics(self, estim):
        mt = estim.compute_metrics(name=self.metrics, to_scalar=False)[self.metrics]
        mt_val = mt.detach().item()
        target_val = self.target_val
        if target_val is None:
            target_val = mt_val
        target_val += self.target_decay * (mt_val - target_val)
        self.target_val = target_val
        return mt


@register
class AddMetricsLoss(AggMetricsLoss):
    """Compute loss by adding Metrics value."""

    def __init__(
        self,
        metrics,
        target_val=None,
        target_decay=0.1,
        lamd=0.01,
    ):
        super().__init__(metrics, target_val, target_decay)
        self.lamd = lamd

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        mt = self._get_metrics(estim)
        return loss + self.lamd * (mt.to(device=loss.device) / self.target_val - 1.)


@register
class MultMetricsLoss(AggMetricsLoss):
    """Compute loss by multiplying Metrics value."""

    def __init__(self, metrics, target_val=None, target_decay=0.1, alpha=1., beta=0.6):
        super().__init__(metrics, target_val, target_decay)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        mt = self._get_metrics(estim)
        return self.alpha * loss * (mt.to(device=loss.device) / self.target_val)**self.beta


@register
class MultLogMetricsLoss(AggMetricsLoss):
    """Compute loss by multiplying logarithm of Metrics value."""

    def __init__(self, metrics, target_val=None, target_decay=0.1, alpha=1., beta=0.6):
        super().__init__(metrics, target_val, target_decay)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, loss, estim, y_pred, X, y_true):
        """Return loss."""
        mt = self._get_metrics(estim)
        return self.alpha * loss * (torch.log(mt.to(device=loss.device)) / math.log(self.target_val))**self.beta


register(torch_criterion_wrapper(CrossEntropyLabelSmoothingLoss))

module = torch.nn

for name, attr in module.__dict__.items():
    if name.startswith('__'):
        continue
    if not callable(attr):
        continue
    if 'Loss' not in name:
        continue
    register(torch_criterion_wrapper(attr))
