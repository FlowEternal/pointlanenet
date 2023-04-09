# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Distributed Estimator."""
from ..base import EstimBase
from modnas.registry.estim import register, build
from modnas.registry.dist_remote import build as build_remote
from modnas.registry.dist_worker import build as build_worker


@register
class DistributedEstim(EstimBase):
    """Distributed Estimator class."""

    def __init__(self, estim_conf, remote_conf, worker_conf, *args, close_remote=True, return_res=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_remote = close_remote
        self.return_res = return_res
        self.is_main = self.config.get('main', False)
        estim_comp_keys = [
            'expman',
            'constructor',
            'exporter',
            'model',
            'writer',
        ]
        estim_comp = {k: getattr(self, k) for k in estim_comp_keys}
        self.estim = build(estim_conf, config=estim_conf, **estim_comp)
        self.estim_step = self.estim.step
        self.estim.step = self.step
        if self.is_main:
            self.remote = build_remote(remote_conf)
        else:
            self.worker = build_worker(worker_conf)

    def step(self, params):
        """Return evaluation results from remote Estimator."""
        def on_done(ret):
            self.logger.debug('Dist main: params: {} ret: {}'.format(params, ret))
            self.estim.step_done(params, ret, self.estim.get_arch_desc())

        def on_failed(ret):
            self.estim.step_done(params, 0, 0)

        if self.is_main:
            self.remote.call('step', params, on_done=on_done, on_failed=on_failed)
            return
        ret = self.estim_step(params)
        self.logger.info('Dist worker: params: {} ret: {}'.format(params, ret))
        return ret

    def run(self, optim):
        """Run Estimator routine."""
        if self.is_main:
            ret = self.estim.run(optim)
            if self.close_remote:
                self.remote.close()
            self.logger.info('Dist main: estim ret: {}'.format(ret))
            if self.return_res:
                return {'main_results': ret}
        else:
            ret = self.worker.run(self.estim)
            self.logger.info('Dist worker: estim ret: {}'.format(ret))
            if self.return_res:
                return {'worker_results': ret}
