# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Report."""
import json
import logging
import os
import glob
import pickle
import time
import random
from copy import deepcopy
import numpy as np
import pandas as pd
from threading import Lock
from collections import OrderedDict
from threading import Thread

import vega
from vega.common import FileOps, TaskOps
from vega.common.general import General
from .record import ReportRecord
from .report_persistence import ReportPersistence
from vega.common import MessageServer
from vega.common.utils import singleton
from vega.common.pareto_front import get_pareto_index

__all__ = ["ReportServer"]
logger = logging.getLogger(__name__)
_records_lock = Lock()
_modified = False


@singleton
class ReportServer(object):
    """Report server."""

    def __init__(self):
        self._hist_records = OrderedDict()
        self.persistence = ReportPersistence()
        self._start_save_report_thread()

    def run(self):
        """Run report server."""
        MessageServer().register_handler("update_record", update_record)
        MessageServer().register_handler("get_record", get_record)

    @property
    def all_records(self):
        """Get all records."""
        return deepcopy(list(self._hist_records.values()))

    def print_best(self, step_name):
        """Print best performance and desc."""
        records = self.get_pareto_front_records(step_name)
        return [dict(worker_id=record.worker_id, performance=record._performance) for record in records]

    def pareto_front(self, step_name=None, nums=None, records=None):
        """Get parent front. pareto."""
        if records is None:
            records = self.all_records
            records = list(filter(lambda x: x.step_name == step_name and x.performance is not None, records))
        records = [record for record in records if record.rewards_compeleted]
        if not records:
            return None, None
        try:
            rewards = [record.rewards if isinstance(record.rewards, list) else [record.rewards] for record in records]
            indexes = get_pareto_index(np.array(rewards)).tolist()
            return [record for i, record in enumerate(records) if indexes[i]]
        except Exception as ex:
            logging.error('No pareto_front_records found, ex=%s', ex)
            return []

    def get_step_records(self, step_name=None):
        """Get step records."""
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps, records))
        return records

    def get_record(self, step_name, worker_id):
        """Get records by step name and worker id."""
        records = self.all_records
        records = list(filter(lambda x: x.step_name == step_name and x.worker_id == worker_id, records))
        return records[0]

    def get_last_record(self):
        """Get last records."""
        if not self.all_records:
            return None
        return self.all_records[-1]

    def get_pareto_front_records(self, step_name=None, nums=None, selected_key=None, choice=None):
        """Get Pareto Front Records."""
        if not step_name:
            step_name = General.step_name
        records = self.all_records
        if selected_key is not None:
            new_records = []
            selected_key.sort()
            for record in records:
                record._objective_keys.sort()
                if record._objective_keys == selected_key:
                    new_records.append(record)
            records = new_records
        filter_steps = [step_name] if not isinstance(step_name, list) else step_name
        records = list(filter(lambda x: x.step_name in filter_steps and x.performance is not None, records))
        if records:
            not_finished = [x.worker_id for x in records if not x.rewards_compeleted]
            records = [x for x in records if x.rewards_compeleted]
            if not_finished:
                logging.info(f"waiting for the workers {str(not_finished)} to finish")
        if not records:
            return []
        pareto = self.pareto_front(step_name, nums, records=records)
        if not pareto:
            return []
        if choice is not None:
            return [random.choice(pareto)]
        else:
            return pareto

    # def _select_one_record(self, outs, choice='normal'):
    #     """Select one record."""
    #     if outs.size == 1:
    #         return outs.astype(int).tolist()
    #     if choice == 'normal':
    #         data = outs[:, 1:].reshape(-1, 1).tolist()
    #         prob = [round(np.log(i + 1e-2), 2) for i in range(1, len(data[0]) + 1)]
    #         prob_temp = prob
    #         for idx, out in enumerate(data):
    #             sorted_ind = np.argsort(out)
    #             for idx, ind in enumerate(sorted_ind):
    #                 prob[ind] += prob_temp[idx]
    #         normalization = [float(i) / float(sum(prob)) for i in prob]
    #         return [np.random.choice(len(data[0]), p=normalization)]

    @classmethod
    def restore(cls):
        """Transfer cvs_file to records."""
        step_path = TaskOps().step_path
        _file = os.path.join(step_path, ".reports")
        if os.path.exists(_file):
            with open(_file, "rb") as f:
                data = pickle.load(f)
            cls._hist_records = data[0]
            cls.__instances__ = data[1]

    def backup_output_path(self):
        """Back up output to local path."""
        backup_path = TaskOps().backup_base_path
        if backup_path is None:
            return
        FileOps.copy_folder(TaskOps().local_output_path, backup_path)

    def output_pareto_front(self, step_name):
        """Save one records."""
        logging.debug("All records in report, records={}".format(self.all_records))
        records = deepcopy(self.get_pareto_front_records(step_name))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump pareto front records, report is emplty.")
            return
        self._output_records(step_name, records)

    def output_step_all_records(self, step_name):
        """Output step all records."""
        records = self.all_records
        logging.debug("All records in report, records={}".format(self.all_records))
        records = list(filter(lambda x: x.step_name == step_name, records))
        logging.debug("Filter step records, records={}".format(records))
        if not records:
            logging.warning("Failed to dump records, report is emplty.")
            return
        self._output_records(step_name, records)

    def _output_records(self, step_name, records):
        """Dump records."""
        columns = ["worker_id", "performance", "desc"]
        outputs = []
        for record in records:
            record = record.serialize()
            _record = {}
            for key in columns:
                _record[key] = record[key]
            outputs.append(deepcopy(_record))
        data = pd.DataFrame(outputs)
        step_path = FileOps.join_path(TaskOps().local_output_path, step_name)
        FileOps.make_dir(step_path)
        _file = FileOps.join_path(step_path, "output.csv")
        try:
            data.to_csv(_file, index=False)
        except Exception:
            logging.error("Failed to save output file, file={}".format(_file))
        for record in outputs:
            worker_id = record["worker_id"]
            worker_path = TaskOps().get_local_worker_path(step_name, worker_id)
            outputs_globs = []
            outputs_globs += glob.glob(FileOps.join_path(worker_path, "desc_*.json"))
            outputs_globs += glob.glob(FileOps.join_path(worker_path, "hps_*.json"))
            outputs_globs += glob.glob(FileOps.join_path(worker_path, "model_*"))
            outputs_globs += glob.glob(FileOps.join_path(worker_path, "performance_*.json"))
            for _file in outputs_globs:
                if os.path.isfile(_file):
                    FileOps.copy_file(_file, step_path)
                elif os.path.isdir(_file):
                    FileOps.copy_folder(_file, FileOps.join_path(step_path, os.path.basename(_file)))

    def set_step_names(self, step_names):
        """Add step information."""
        global _records_lock, _modified
        with _records_lock:
            _modified = True
            self.persistence.set_step_names(step_names)

    def update_step_info(self, **kwargs):
        """Update step information."""
        global _records_lock, _modified
        with _records_lock:
            _modified = True
            self.persistence.update_step_info(**kwargs)

    def __repr__(self):
        """Override repr function."""
        return str(self.all_records)

    @classmethod
    def load_records_from_model_folder(cls, model_folder):
        """Transfer json_file to records."""
        if not model_folder or not os.path.exists(model_folder):
            logging.error("Failed to load records from model folder, folder={}".format(model_folder))
            return []
        records = []
        pattern = FileOps.join_path(model_folder, "desc_*.json")
        files = glob.glob(pattern)
        for _file in files:
            try:
                with open(_file) as f:
                    worker_id = _file.split(".")[-2].split("_")[-1]
                    weights_file = os.path.join(os.path.dirname(_file), "model_{}".format(worker_id))
                    if vega.is_torch_backend():
                        weights_file = '{}.pth'.format(weights_file)
                    elif vega.is_ms_backend():
                        weights_file = '{}.ckpt'.format(weights_file)
                    if not os.path.exists(weights_file):
                        weights_file = None

                    sample = dict(worker_id=worker_id, desc=json.load(f), weights_file=weights_file)
                    record = ReportRecord().load_dict(sample)
                    records.append(record)
            except Exception as ex:
                logging.info('Can not read records from json because {}'.format(ex))
        return records

    def _start_save_report_thread(self):
        _thread = Thread(target=_dump_report, args=(self, self.persistence,))
        _thread.daemon = True
        _thread.start()


def update_record(step_name=None, worker_id=None, **kwargs):
    """Update record."""
    if step_name is None or worker_id is None:
        return {"result": "failed", "message": "request message missing step_name or worker id."}
    if kwargs:
        kwargs["step_name"] = step_name
        kwargs["worker_id"] = worker_id
        uid = "{}_{}".format(step_name, worker_id)
        global _records_lock, _modified
        with _records_lock:
            _modified = True
            records = ReportServer()._hist_records
            if uid in records:
                records[uid].load_dict(kwargs)
                logging.debug("update record: {}".format(records[uid].to_dict()))
            else:
                records[uid] = ReportRecord().load_dict(kwargs)
                logging.debug("new record: {}".format(records[uid].to_dict()))
    return {"result": "success", "data": records[uid].to_dict()}


def get_record(step_name=None, worker_id=None, **kwargs):
    """Get record."""
    if step_name is None or worker_id is None:
        return {"result": "failed", "message": "require message missing step_name or worker id."}
    uid = "{}_{}".format(step_name, worker_id)
    records = ReportServer()._hist_records
    if uid in records:
        data = records[uid].to_dict()
    else:
        data = ReportRecord().to_dict()
    return {"result": "success", "data": data}


def _dump_report(report_server, persistence):
    while True:
        time.sleep(1)
        global _records_lock, _modified
        with _records_lock:
            if not _modified:
                continue
            all_records = deepcopy(report_server.all_records)
            _modified = False

        try:
            persistence.save_report(all_records)
            # TODO
            # persistence.pickle_report(report_server._hist_records, report_server.__instances__)
            report_server.backup_output_path()
        except Exception as e:
            logging.warning(f"Failed to dump reports, message={str(e)}")
