# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Fully Train PipeStep that used in Pipeline."""
import logging
from vega.common.general import General
from vega.common.class_factory import ClassFactory, ClassType
from vega.report import ReportServer, ReportRecord, ReportClient
from vega.common import Status
from vega.core.scheduler import create_master
from vega.core.pipeline.conf import PipeStepConfig
from vega.core.pipeline.train_pipe_step import TrainPipeStep
from vega.trainer.conf import TrainerConfig


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.PIPE_STEP)
class MultiTaskPipeStep(TrainPipeStep):
    """TrainPipeStep is the implementation class of PipeStep.

    Fully train is the last pipe step in pipeline, we provide horovrd or local trainer
    for user to choose.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distributed_training = not General._parallel and TrainerConfig.distributed
        logger.info("init MultiTaskPipeStep...")

    def do(self):
        """Start to run fully train with horovod or local trainer."""
        logger.info("MultiTaskPipeStep started...")
        self.update_status(Status.running)
        self.master = create_master()
        self._train_multi_task()
        self.master.join()
        ReportServer().output_step_all_records(step_name=self.task.step_name)
        self.master.close()
        ReportServer().backup_output_path()
        self.update_status(Status.finished)

    def _train_single_model(self, model_desc, model_id, hps, multi_task):
        cls_trainer = ClassFactory.get_cls(ClassType.TRAINER, PipeStepConfig.trainer.type)
        step_name = self.task.step_name
        sample = dict(worker_id=model_id, desc=model_desc, step_name=step_name)
        record = ReportRecord().load_dict(sample)
        logging.debug("update record=%s", str(record))
        trainer = cls_trainer(model_desc=model_desc, id=model_id, hps=hps, multi_task=multi_task)
        ReportClient().update(**record.to_dict())
        if self._distributed_training:
            self._do_distributed_fully_train(trainer)
        else:
            self._do_single_fully_train(trainer)

    def _train_multi_task(self):
        from copy import deepcopy
        for epoch in range(0, PipeStepConfig.pipe_step.multi_task_epochs):
            for alg in PipeStepConfig.pipe_step.tasks:
                desc = deepcopy(PipeStepConfig().to_dict()[alg])
                model_desc = desc.model.model_desc
                desc.pop('model')
                self._train_single_model(model_desc=model_desc, model_id=0, hps=desc, multi_task=alg)
