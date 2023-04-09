# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Calculate model forward latency."""
import time
import vega
import numpy as np
import os
from vega.evaluator.conf import DeviceEvaluatorConfig
import datetime
import logging


def calc_forward_latency(model, input, sess_config=None, num=10):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :return: forward latency
    :rtype: float
    """
    if DeviceEvaluatorConfig.remote_host:
        return _calc_forward_latency_davinci(model, input, sess_config, num, DeviceEvaluatorConfig().to_dict())
    return _calc_forward_latency_gpu(model, input, sess_config, num)


def _calc_forward_latency_gpu(model, input, sess_config=None, num=100):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :return: forward latency
    :rtype: float
    """
    prepare_num = int(0.05 * num)
    if vega.is_torch_backend():
        pre_mode = model.training
        model.train(False)
        for _ in range(prepare_num):
            model(input)
        start_time = time.time()
        for _ in range(num):
            model(input)
        latency = (time.time() - start_time) / num
        model.train(pre_mode)
    elif vega.is_tf_backend():
        import tensorflow.compat.v1 as tf
        with tf.Graph().as_default():
            input_holder = tf.placeholder(dtype=tf.float32, shape=input.shape.as_list())
            pre_mode = model.training
            model.training = False
            output = model(input_holder)
            with tf.Session(config=sess_config) as sess:
                sess.run(tf.global_variables_initializer())
                input = tf.random.uniform(input.shape.as_list(), dtype=input.dtype)
                input_numpy = input.eval(session=sess)
                for _ in range(prepare_num):
                    sess.run(output, feed_dict={input_holder: input_numpy})
                start_time = time.time()
                for _ in range(num):
                    sess.run(output, feed_dict={input_holder: input_numpy})
                latency = (time.time() - start_time) / num
            model.training = pre_mode
    elif vega.is_ms_backend():
        latency = 0.
    return latency


def _calc_forward_latency_davinci(model, input, sess_config=None, num=10, evaluate_config=None):
    """Model forward latency calculation.

    :param model: network model
    :type model: torch or tf module
    :param input: input tensor
    :type input: Tensor of torch or tf
    :param num: forward number
    :type num: int
    :param evaluate_config: some config for evaluate in davinci
    :type evaluate_config: dict
    :return: forward latency
    :rtype: float
    """
    from vega.evaluator.tools.evaluate_davinci_bolt import evaluate
    from vega.common.task_ops import TaskOps
    # backend = evaluate_config.get("backend")
    hardware = evaluate_config.get("hardware")
    remote_host = evaluate_config.get("remote_host")
    worker_path = TaskOps().local_base_path
    save_data_file = os.path.join(worker_path, "input.bin")

    latency = 0.
    now_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    job_id = "pre_evaluate_" + now_time
    logging.info("The job id of evaluate service is {}.".format(job_id))
    if vega.is_torch_backend():
        import torch
        input_shape = input.shape
        if torch.is_tensor(input):
            input = input.cpu().numpy()
        input.tofile(save_data_file)
        for index in range(num):
            reuse_model = False if index == 0 else True
            results = evaluate("pytorch", hardware, remote_host, model, None, save_data_file, input_shape,
                               reuse_model, job_id)
            latency += np.float(results.get("latency"))
    elif vega.is_tf_backend():
        input_shape = input.shape.as_list()
        test_data = np.random.random(input_shape).astype(np.float32)
        test_data.tofile(save_data_file)
        for index in range(num):
            reuse_model = False if index == 0 else True
            results = evaluate("tensorflow", hardware, remote_host, model, None, save_data_file, input_shape,
                               reuse_model, job_id)
            latency += np.float(results.get("latency"))
    return latency / num
