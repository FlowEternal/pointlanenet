# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model zoo."""
import vega
import logging
import glob
import numpy
from collections import OrderedDict
import os
from vega.networks.network_desc import NetworkDesc
from vega.common.general import General
from vega.modules.graph_utils import graph2desc
from vega.modules.module import Module
from vega.modules.arch import transform_architecture


class ModelZoo(object):
    """Model zoo."""

    @classmethod
    def set_location(cls, location):
        """Set model zoo location.

        :param location: model zoo location.
        :type localtion: str.

        """
        General.model_zoo.model_zoo_path = location

    @classmethod
    def get_model(cls, model_desc=None, pretrained_model_file=None, exclude_weight_prefix=None):
        """Get model from model zoo.

        :param network_name: the name of network, eg. ResNetVariant.
        :type network_name: str or None.
        :param network_desc: the description of network.
        :type network_desc: str or None.
        :param pretrained_model_file: path of model.
        :type pretrained_model_file: str.
        :return: model.
        :rtype: model.

        """
        model = None
        if model_desc is not None:
            try:
                network = NetworkDesc(model_desc)
                model = network.to_model()
            except Exception as e:
                logging.error("Failed to get model, model_desc={}, msg={}".format(model_desc, str(e)))
                raise e
        logging.info("Model was created.")
        if not isinstance(model, Module):
            model = cls.to_module(model)
        if pretrained_model_file is not None:
            if exclude_weight_prefix:
                model.exclude_weight_prefix = exclude_weight_prefix
            model = cls._load_pretrained_model(model, pretrained_model_file)
        model = transform_architecture(model, pretrained_model_file)
        if model is None:
            raise ValueError("Failed to get mode, model is None.")
        return model

    @classmethod
    def to_module(cls, model):
        """Build model desc before get model."""
        if vega.is_ms_backend():
            from vega.networks.mindspore.backbones.ms2vega import transform_model
            return transform_model(model)
        else:
            try:
                model_desc = cls.parse_desc_from_pretrained_model(model)
            except Exception as ex:
                logging.warn("Parse model desc failed: {}".format(ex))
                return model
            return ModelZoo.get_model(model_desc)

    @classmethod
    def parse_desc_from_pretrained_model(cls, src_model, pb_file=None):
        """Parse desc from Petrained Model."""
        import tensorflow.compat.v1 as tf
        from tensorflow.python.framework import tensor_util
        tf.reset_default_graph()
        data_shape = (1, 224, 224, 3)
        x = tf.ones(data_shape)
        if pb_file:
            weights = OrderedDict()
            with tf.io.gfile.GFile(pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.Graph()
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
            wts = [n for n in graph_def.node if n.op == 'Const']
            for n in wts:
                weights[n.name] = tensor_util.MakeNdarray(n.attr['value'].tensor)
        else:
            src_model(x, False)
            graph = tf.get_default_graph()
        desc = graph2desc(graph)
        tf.reset_default_graph()
        return desc

    @classmethod
    def _load_pretrained_model(cls, model, pretrained_model_file):
        pretrained_model_file = cls._get_abs_path(pretrained_model_file)
        logging.info("load model weights from file, weights file={}".format(pretrained_model_file))
        if vega.is_torch_backend():
            if not os.path.isfile(pretrained_model_file):
                raise Exception(f"Pretrained model is not existed, model={pretrained_model_file}")
            import torch
            checkpoint = torch.load(pretrained_model_file)
            model.load_state_dict(checkpoint,strict=False) # TODO strict

            # del checkpoint
        if vega.is_tf_backend():
            if pretrained_model_file.endswith('.pth'):
                checkpoint = convert_checkpoint_from_pytorch(pretrained_model_file, model)
                model.load_checkpoint_from_numpy(checkpoint)
            else:
                pretrained_model_file = cls._get_tf_model_file(pretrained_model_file)
                model.load_checkpoint(pretrained_model_file)
        elif vega.is_ms_backend():
            from mindspore.train.serialization import load_checkpoint
            if hasattr(model, "pretrained"):
                pretrained_weight = model.pretrained(pretrained_model_file)
            else:
                if os.path.isfile(pretrained_model_file):
                    pretrained_weight = pretrained_model_file
                else:
                    for file in os.listdir(pretrained_model_file):
                        if file.endswith(".ckpt"):
                            pretrained_weight = os.path.join(pretrained_model_file, file)
                            break
            load_checkpoint(pretrained_weight, net=model)
            # os.remove(pretrained_weight)
        return model

    @classmethod
    def select_compressed_models(cls, model_zoo_file, standard, num):
        """Select compressed model by model filter."""
        from vega.model_zoo.compressed_model_filter import CompressedModelFilter
        model_filter = CompressedModelFilter(model_zoo_file)
        model_desc_list = model_filter.select_satisfied_model(standard, num)
        return model_desc_list

    @classmethod
    def _get_abs_path(cls, _path):
        if "{local_base_path}" in _path:
            from vega.common.task_ops import TaskOps
            return os.path.abspath(_path.replace("{local_base_path}", TaskOps().local_base_path))
        return _path

    @classmethod
    def _get_tf_model_file(cls, _path):
        if _path.endswith(".pb"):
            return _path
        pts = glob.glob(_path + "/*.pb")
        if pts:
            return pts[0]
        ckpt1 = glob.glob(_path + "/checkpoint")
        ckpt2 = glob.glob(_path + "/*.data-*")
        ckpt3 = glob.glob(_path + "/*.meta")
        ckpt4 = glob.glob(_path + "/*.index")
        if ckpt1 and ckpt2 and ckpt3 and ckpt4:
            return ckpt3[0][:-5]
        subpaths = glob.glob(_path + "/*")
        for subpath in subpaths:
            if os.path.isdir(subpath):
                ckpt1 = glob.glob(subpath + "/checkpoint")
                ckpt2 = glob.glob(subpath + "/*.data-*")
                ckpt3 = glob.glob(subpath + "/*.meta")
                ckpt4 = glob.glob(subpath + "/*.index")
                if ckpt1 and ckpt2 and ckpt3 and ckpt4:
                    return ckpt3[0][:-5]
        return _path


def convert_checkpoint_from_pytorch(pretrained_model_file, model):
    """Convert checkpoint from pytorch."""
    logging.info("Convert model weights from pytorch file, weights file={}".format(pretrained_model_file))
    import torch
    states = torch.load(pretrained_model_file)
    checkpoint = OrderedDict()
    named_modules = model.named_modules()
    for name, state in states.items():
        if 'num_batches_tracked' in name:
            continue
        scope_name = name[::-1].replace('.', '/', 1)[::-1]
        scope_name = scope_name.split('/')[0] + '/'
        for module_name, module in named_modules:
            if not module_name.startswith(scope_name):
                continue
            tf_name, transpose = convert_pytorch_name_to_tf(name, module_name)
            if transpose:
                state = numpy.transpose(state)
            checkpoint[tf_name] = state
    return checkpoint


def convert_pytorch_name_to_tf(torch_name, module_name=None):
    """Convert a pytorch weight name in a tensorflow model weight name."""
    op_name = torch_name.split(".")[-1]
    if op_name == "weight":
        op_name = "gamma" if module_name and '/BatchNorm2d' in module_name else 'kernel'
    transpose = bool(op_name == "kernel" or "emb_projs" in op_name or "out_projs" in op_name)
    if op_name == "bias":
        op_name = "bias" if module_name and '/Linear' in module_name else 'beta'
    if op_name == "running_mean":
        op_name = "moving_mean"
    if op_name == "running_var":
        op_name = "moving_variance"
    return module_name + '/' + op_name, transpose
