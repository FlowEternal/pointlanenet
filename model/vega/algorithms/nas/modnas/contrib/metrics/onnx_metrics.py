# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ONNX export metrics."""
import os
import tempfile
import torch
from modnas.registry.metrics import build, register
from modnas.metrics.base import MetricsBase


@register
class OnnxExportMetrics(MetricsBase):
    """ONNX export metrics class."""

    def __init__(self, metrics, head=None, export_dir=None, verbose=False):
        super().__init__()
        self.metrics = build(metrics)
        if head is None:
            head = 'name'
        self.head = head
        if export_dir is None:
            export_dir = tempfile.gettempdir()
        os.makedirs(export_dir, exist_ok=True)
        self.export_dir = export_dir
        self.verbose = verbose
        self.exported = {}

    def __call__(self, node):
        """Return metrics output."""
        key = '#'.join([str(node[k]) for k in self.head if node[k] is not None])
        onnx_info = self.exported.get(key, None)
        if onnx_info is not None:
            return self.metrics(onnx_info)
        in_shape = node['in_shape']
        module = node.module
        plist = list(module.parameters())
        device = None if len(plist) == 0 else plist[0].device
        export_dir = os.path.join(self.export_dir, key)
        os.makedirs(export_dir, exist_ok=True)
        model_path = os.path.join(export_dir, 'model.onnx')
        dummy_input = torch.randn(in_shape).to(device=device)
        input_names = ['input']
        input_shapes = [tuple(dummy_input.shape)]
        output_names = ['output']
        output_shapes = [tuple()]
        with torch.no_grad():
            torch.onnx.export(module,
                              dummy_input,
                              model_path,
                              verbose=self.verbose,
                              input_names=input_names,
                              output_names=output_names)
        onnx_info = {
            'model_path': model_path,
            'input_names': input_names,
            'input_shapes': input_shapes,
            'output_names': output_names,
            'output_shapes': output_shapes
        }
        self.exported[key] = onnx_info
        ret = self.metrics(onnx_info)
        self.logger.info('onnx export: {}: {}'.format(key, ret))
        return ret
