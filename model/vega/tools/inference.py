# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""

import pickle
import os
import numpy as np
import cv2
import csv
import vega
from vega.common import argment_parser


def _load_data(args):
    """Load data from path."""
    if not os.path.exists(args.data_path):
        raise("data path is empty, path={}".format(args.data_path))
    else:
        _path = os.path.abspath(args.data_path)
        _files = [(os.path.join(_path, _file)) for _file in os.listdir(_path)]
        _files = [_file for _file in _files if os.path.isfile(_file)]
        return _files


def _load_image(image_file):
    """Load every image."""
    img = cv2.imread(image_file)
    img = img / 255
    img = img.astype(np.float32)
    width, height, channel = img.shape
    return img.reshape(1, channel, height, width)


def _to_tensor(data):
    """Change data to tensor."""
    if vega.is_torch_backend():
        import torch
        data = torch.tensor(data)
        if args.device == "GPU":
            return data.cuda()
        else:
            return data
    elif vega.is_tf_backend():
        import tensorflow as tf
        data = tf.convert_to_tensor(data)
        return data


def _get_model(args):
    """Get model."""
    from vega.model_zoo import ModelZoo
    model = ModelZoo.get_model(args.model_desc, args.model)
    if vega.is_torch_backend():
        if args.device == "GPU":
            model = model.cuda()
        model.eval()
    return model


def _infer(args, loader, model=None):
    """Choose backend."""
    if vega.is_torch_backend():
        return _infer_pytorch(model, loader)
    elif vega.is_tf_backend():
        return _infer_tf(args, model, loader)
    elif vega.is_ms_backend():
        return _infer_ms(args, model, loader)


def _infer_pytorch(model, loader):
    """Infer with pytorch."""
    infer_result = []
    import torch
    with torch.no_grad():
        for file_name in loader:
            data = _to_tensor(_load_image(file_name))
            logits = model(data)
            logits = logits[0].tolist()
            infer_result.append((os.path.basename(file_name), logits))
        return infer_result


def _infer_tf(args, model, loader):
    """Infer with tf."""
    infer_result = []
    import tensorflow as tf
    meta_file = None
    model_file = os.listdir(args.model)
    for file in model_file:
        if '.meta' in file:
            meta_file = file
    _, channel, height, width = _load_image(loader[0]).shape
    input = tf.placeholder(tf.float32, shape=(1, channel, height, width), name='input')
    output = model(input, training=False)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if meta_file is not None:
            saver = tf.train.import_meta_graph(args.model + '/{}'.format(meta_file))
            saver.restore(sess, tf.train.latest_checkpoint(args.model))
        else:
            print('Meta file cant find')
            raise
        for filename in loader:
            data = _load_image(filename)
            infer_result.append(sess.run(output, feed_dict={input: data}))
    return infer_result


def _infer_ms():
    """Infer with ms."""
    # TODO
    pass


def _save_result(args, result):
    """Save results."""
    _output_file = args.output_file
    if args.data_format in ["classification", "c"]:
        if not _output_file:
            _output_file = "./result.csv"
        result = [(_file, _data.index(max(_data))) for (_file, _data) in result]
        with open(_output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(result)
        print('Results of Inference is saved in {}.'.format(_output_file))
    else:
        if not _output_file:
            _output_file = "./result.pkl"
        with open(_output_file, 'wb') as f:
            pickle.dump(result, f)
        print('Results of Inference is saved in {}.'.format(_output_file))


def parse_args_parser():
    """Parse parameters."""
    parser = argment_parser('Vega Inference.')
    parser.add_argument("-c", "--model_desc", default=None, type=str, required=True,
                        help="model description file, generally in json format, contains 'module' node.")
    parser.add_argument("-m", "--model", default=None, type=str, required=True,
                        help="model weight file, usually ends with pth, ckpl, etc.")
    parser.add_argument("-df", "--data_format", default="classification", type=str,
                        choices=["classification", "c",
                                 "super_resolution", "s",
                                 "segmentation", "g",
                                 "detection", "d"],
                        help="data type, "
                        "classification: some pictures file in a folder, "
                        "super_resolution: some low resolution picture in a folder, "
                        "segmentation: , "
                        "detection: . "
                        "'classification' is default"
                        )
    parser.add_argument("-dp", "--data_path", default=None, type=str, required=True,
                        help="the folder where the file to be inferred is located.")
    parser.add_argument("-b", "--backend", default="pytorch", type=str,
                        choices=["pytorch", "tensorflow", "mindspore"],
                        help="set training platform")
    parser.add_argument("-d", "--device", default="GPU", type=str,
                        choices=["CPU", "GPU", "NPU"],
                        help="set training device")
    parser.add_argument("-o", "--output_file", default=None, type=str,
                        help="output file. "
                        "classification: ./result.csv, "
                        "super_resolution: ./result.pkl, "
                        "segmentation: ./result.pkl, "
                        "detection: ./result.pkl "
                        )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_parser()
    vega.set_backend(args.backend, args.device)
    print("Start building model.")
    model = _get_model(args)
    print("Start loading data.")
    loader = _load_data(args)
    print("Start inferencing.")
    result = _infer(args, loader, model)
    _save_result(args, result)
    print("Completed successfully.")
