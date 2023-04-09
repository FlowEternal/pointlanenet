# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
from vega.common import ClassType
from vega.common import ConfigSerializable
from vega.networks.model_config import ModelConfig
from vega.trainer.conf import TrainerConfig
from vega.evaluator.conf import EvaluatorConfig
from vega.datasets.conf.dataset import DatasetConfig
from vega.common.check import valid_rule


class SearchSpaceConfig(ConfigSerializable):
    """Default Search Space config for Pipeline."""

    # _type_name = ClassType.NETWORK
    type = None

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(SearchSpaceConfig, cls).from_dict(data, skip_check)
        if "type" in data and not data["type"]:
            if hasattr(cls, "hyperparameters"):
                del cls.hyperparameters
        return cls

    @classmethod
    def check_config(cls, config):
        """Check config."""
        check_rules_searchspace = {"type": {"type": str},
                                   "modules": {"type": list}
                                   }
        valid_rule(cls, config, check_rules_searchspace)
        # to do
        # for module in config["modules"]:
        #    if module not in config:
        #        raise Exception(
        #            "{} is required in {}".format(module, cls.__name__))


class SearchAlgorithmConfig(ConfigSerializable):
    """Default Search Algorithm config for Pipeline."""

    type = None
    _class_type = ClassType.SEARCH_ALGORITHM
    _class_data = None

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(SearchAlgorithmConfig, cls).from_dict(data, skip_check)
        if "type" in data and not data["type"]:
            cls._class_data = None
            cls.type = None
        return cls

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_searchalgorithm = {
            "type": {"required": True, "type": str}}
        return rules_searchalgorithm


class PipeStepConfig(ConfigSerializable):
    """Default Pipeline config for Pipe Step."""

    type = "SearchPipeStep"
    dataset = DatasetConfig
    search_algorithm = SearchAlgorithmConfig
    search_space = SearchSpaceConfig
    model = ModelConfig
    trainer = TrainerConfig
    evaluator = EvaluatorConfig
    evaluator_enable = False
    models_folder = None
    pipe_step = {}

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        cls = super(PipeStepConfig, cls).from_dict(data, skip_check)
        if "pipe_step" in data:
            if "type" in data["pipe_step"]:
                cls.type = data["pipe_step"]["type"]
            if "models_folder" in data["pipe_step"]:
                cls.models_folder = data["pipe_step"]["models_folder"]
        return cls

    @classmethod
    def check_config(cls, config):
        """Check config."""
        check_rules_nas = {"pipe_step": {"type": dict},
                           "dataset": {"type": dict},
                           "search_algorithm": {"type": dict},
                           "search_space": {"type": dict},
                           "trainer": {"type": dict},
                           "evaluator": {"type": dict},
                           "model": {"type": dict},
                           "evaluator_enable": {"type": bool},
                           "models_folder": {"type": str}
                           }

        if 'pipe_step' in config:
            if config["pipe_step"]['type'] == 'SearchPipeStep':
                valid_rule(cls, config, check_rules_nas)
            elif config["pipe_step"]['type'] == 'TrainPipeStep':
                pass
        else:
            raise Exception(
                "pipe_step attr is required in {}".format(cls.__name__))


class PipelineConfig(ConfigSerializable):
    """Pipeline config."""

    steps = []
