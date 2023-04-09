# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""
from .modules.conf.loss import LossConfig
from .modules.conf.lr_scheduler import LrSchedulerConfig
from .modules.conf.optim import OptimConfig
from vega.common import ConfigSerializable
from .task_conf import DEFAULT_CONFIG


class MetricsConfig(ConfigSerializable):
    """Default Metrics Config."""

    _class_type = "trainer.metric"
    _update_all_attrs = True
    type = 'accuracy'
    params = {}

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(MetricsConfig, cls).from_dict(data, skip_check)
        if "params" not in data:
            cls.params = {}
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        check_rules = {"type": {"type": str},
                       "params": {"type": dict}}
        return check_rules


class TrainerConfig(ConfigSerializable):
    """Default Trainer Config."""

    type = 'Trainer'
    task = 'Classification'
    train_lane = None # 是否训练车道线
    actions_list = None
    with_valid = True
    with_train = True
    max_train_steps = None
    is_detection_trainer = False
    is_gan_trainer = False
    distributed = False
    save_model_desc = False
    save_ext_model = False
    report_freq = 10
    seed = 0
    epochs = 1
    valid_interval = 1
    syncbn = False
    amp = False
    lazy_built = False
    callbacks = None
    grad_clip = None
    pretrained_model_file = None
    model_statistics = True
    # config a object
    optimizer = OptimConfig.from_dict(DEFAULT_CONFIG[task]['optimizer'])
    lr_scheduler = LrSchedulerConfig.from_dict(DEFAULT_CONFIG[task]['lr_scheduler'])
    metric = MetricsConfig.from_dict(DEFAULT_CONFIG[task]['metric'])
    loss = LossConfig.from_dict(DEFAULT_CONFIG[task]['loss'])
    # TODO: need to delete
    limits = None
    init_model_file = None
    pareto_front_file = None
    unrolled = True
    model_desc_file = None
    codec = None
    model_desc = None
    hps_file = None
    hps_folder = None
    loss_scale = 1.
    save_steps = 500
    report_on_valid = False
    perfs_cmp_mode = None
    perfs_cmp_key = None
    call_metrics_on_train = True
    report_on_epoch = False
    calc_params_each_epoch = False
    model_path = None
    get_train_metric_after_epoch = True
    kwargs = None
    train_verbose = 2
    valid_verbose = 2
    train_report_steps = 10
    valid_report_steps = 10
    load_checkpoint = True
    save_checkpoint = True
    load_weights_file = True
    use_unsupervised_pretrain = False
    calc_latency = False
    train_in_once = False
    mixup = False
    multi_task = False
    adaptive_muti_loss = False

    @classmethod
    def set_task(cls, task):
        """Set default config for trainer."""
        default_conf = DEFAULT_CONFIG[task]
        for key in default_conf.keys():
            if hasattr(cls, key):
                attr = getattr(cls, key)
                setattr(cls, key, attr.from_dict(default_conf[key]))

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        if "task" in data.keys() and data["task"] != cls.task and data["task"] is not None:
            cls.set_task(data["task"])
        return super(TrainerConfig, cls).from_dict(data, skip_check)

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        check_rules_trainer = {"type": {"type": str},
                               "task": {"type": (str, None)},
                               "with_valid": {"type": bool},
                               "is_detection_trainer": {"type": bool},
                               "distributed": {"type": bool},
                               "save_model_desc": {"type": bool},
                               "report_freq": {"type": int},
                               "seed": {"type": int},
                               "epochs": {"type": int},
                               "valid_interval": {"type": int},
                               "syncbn": {"type": bool},
                               "amp": {"type": bool},
                               "lazy_built": {"type": bool},
                               "callbacks": {"type": (list, str, None)},
                               "grad_clip": {"type": (int, float, None)},
                               "pretrained_model_file": {"type": (str, None)},
                               "model_statistics": {"type": bool},
                               "optimizer": {"type": (dict, list)},
                               "lr_scheduler": {"type": dict},
                               "loss": {"type": dict},
                               "metric": {"type": dict},
                               "limits": {"type": (dict, None)},
                               "init_model_file": {"type": (str, None)},
                               "pareto_front_file": {"type": (str, None)},
                               "unrolled": {"type": bool},
                               "model_desc_file": {"type": (str, None)},
                               "codec": {"type": (str, dict, None)},
                               "model_desc": {"type": (str, dict, None)},
                               "hps_file": {"type": (str, None)},
                               "hps_folder": {"type": (str, None)},
                               "loss_scale": {"type": (int, float)},
                               "save_steps": {"type": int},
                               "report_on_valid": {"type": bool},
                               "call_metrics_on_train": {"type": bool},
                               "get_train_metric_after_epoch": {"type": bool},
                               "train_verbose": {"type": int},
                               "valid_verbose": {"type": int},
                               "train_report_steps": {"type": int},
                               "valid_report_steps": {"type": int},
                               "calc_params_each_epoch": {"type": bool},
                               "load_checkpoint": {"type": bool},
                               "mixup": {"type": bool},
                               "multi_task": {"type": bool},
                               "adaptive_muti_loss": {"type": bool}
                               }
        return check_rules_trainer

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {
            "optimizer": cls.optimizer,
            "lr_scheduler": cls.lr_scheduler,
            "metric": cls.metric,
            "loss": cls.loss
        }
