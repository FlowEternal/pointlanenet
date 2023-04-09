# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report callback defination."""
import os
import logging
from .callback import Callback
from vega.report import ReportClient
from vega.common import ClassFactory, ClassType
import vega

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class ReportCallback(Callback):
    """Callback that report records."""

    def __init__(self):
        """Initialize ReportCallback callback."""
        super(ReportCallback, self).__init__()
        self.epoch = 0
        self.priority = 280

    def before_train(self, logs=None):
        """Close the connection of report."""
        self._update_report()

    def after_valid(self, logs=None):
        """Be called after each epoch."""
        if self.trainer.config.report_on_valid:
            self._update_report()

    def after_epoch(self, epoch, logs=None):
        """Be called after each epoch."""
        self.epoch = epoch
        self._update_report(epoch)

    def after_train(self, logs=None):
        """Close the connection of report."""
        record = self._update_report(self.trainer.epochs - 1)
        if hasattr(record, "rung_id"):
            self._next_rung(record)

    def _update_report(self, epoch=0):
        if self.trainer.standalone:
            return
        if self.trainer.distributed and os.environ["DEVICE_ID"] != "0":
            return
        try:
            record = ReportClient().get_record(self.trainer.step_name, self.trainer.worker_id)
        except Exception as e:
            logger.warn(f"failed to update record to report server, message: {e}")
            return
        if hasattr(self.trainer.model, '_arch_params_type') and self.trainer.model._arch_params_type:
            if vega.is_ms_backend():
                record.desc = self.trainer.model_desc
            else:
                record.desc = self.trainer.model.to_desc()
        if not record.desc:
            record.desc = self.trainer.model_desc
        if not record.hps and self.trainer.hps:
            record.hps = self.trainer.hps
        try:
            record = ReportClient().update(
                self.trainer.step_name,
                self.trainer.worker_id,
                desc=record.desc,
                hps=record.hps,
                performance=self.trainer.best_performance or self.trainer.performance,
                objectives=self.trainer.valid_metrics.objectives,
                epoch=self.trainer.epochs,
                current_epoch=epoch + 1,
                num_epochs=self.trainer.epochs,
                model_path=self.trainer.ext_model if self.trainer.ext_model is not None else self.trainer.model_path,
                checkpoint_path=self.trainer.checkpoint_file,
                weights_file=self.trainer.weights_file,
                runtime=self.trainer.runtime,
                multi_task=self.trainer.multi_task,
            )
        except Exception as e:
            logger.warn(f"failed to update record to report server, message: {e}")
            return
        logging.debug("report_callback record: {}".format(record.to_dict()))
        return record

    def _next_rung(self, record):
        if self.trainer.standalone:
            return
        result = ReportClient().request(action="next_rung", **record.to_dict())
        logging.debug(f"next rung result: {result}")

        if not isinstance(result, dict) or "result" not in result or result["result"] != "success":
            self.trainer._next_rung = False
            return
        if result["data"]["rung_id"] is None:
            self.trainer._next_rung = False
            return

        self.trainer._next_rung = True
        self.trainer._start_epoch = self.trainer.epochs
        self.trainer.epochs += int(result["data"]["epochs"])
        ReportClient().update(
            step_name=record.step_name,
            worker_id=record.worker_id,
            rung_id=int(result["data"]["rung_id"]),
            num_epochs=self.trainer.epochs,
        )
