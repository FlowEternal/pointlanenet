# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The trainer program for Auto Lane."""

import logging
from vega.common import ClassFactory, ClassType
from vega.common import FileOps
from vega.trainer.callbacks import Callback

# for validation
from collections import ChainMap
from vega.datasets.common.utils.auto_lane_codec_utils import nms_with_pos, order_lane_x_axis
from vega.datasets.common.utils.auto_lane_codec_utils import convert_lane_to_dict
import ujson
import torch
import os
import numpy as np
import cv2

#---------------------------------------------------#
#  检测评估相关函数
#---------------------------------------------------#
from vega.networks.pytorch.detectors.det_box_coder import BoxCoder
from vega.networks.pytorch.detectors.nms_wrapper import nms
from vega.networks.pytorch.detectors.det_bbox import clip_boxes
from vega.datasets.common.utils.det_bbox import rbox_2_quad
from vega.networks.pytorch.detectors.det.anchors import Anchors
import codecs


DET_CLASS_LIST = ('__background__',
                  "roadtext",
                  "pedestrian",
                  "guidearrow",
                  "traffic",
                  "obstacle",
                  "vehicle_wheel",
                  "roadsign",
                  "vehicle",
                  "vehicle_light"
                  )


logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.CALLBACK)
class AutoLaneTrainerCallback(Callback):
    """Construct the trainer of Auto Lane."""

    disable_callbacks = ['ProgressLogger', 'MetricsEvaluator', "ModelStatistics"]

    def logger_patch(self):
        """Patch the default logger."""
        worker_path = self.trainer.get_local_worker_path()
        worker_spec_log_file = FileOps.join_path(worker_path, 'current_worker.log')
        logger = logging.getLogger(__name__)
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)
        for hdlr in logging.root.handlers:
            logging.root.removeHandler(hdlr)
        logger.addHandler(logging.FileHandler(worker_spec_log_file))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logging.root = logger

    def before_train(self, logs=None):
        """Be called before the whole train process."""
        self.trainer.config.call_metrics_on_train = False
        self.cfg = self.trainer.config
        self.worker_id = self.trainer.worker_id
        self.local_base_path = self.trainer.local_base_path
        self.local_output_path = self.trainer.local_output_path

        self.result_path = FileOps.join_path(self.trainer.local_base_path, "result")

        # 是否训练车道线
        self.train_lane = self.trainer.config.train_lane
        FileOps.make_dir(self.result_path)
        self.logger_patch()

    def make_batch(self, batch):
        """Make batch for each training step."""
        image = batch.pop('image').cuda(non_blocking=True).float()
        return image, batch

    def train_step(self, batch):
        """Replace the default train_step function."""
        self.trainer.model.train()
        image, train_item_spec = batch

        gt_loc = train_item_spec.pop('gt_loc').cuda(non_blocking=True).float()
        gt_cls = train_item_spec.pop('gt_cls').cuda(non_blocking=True).float()

        self.trainer.optimizer.zero_grad()
        model_out = self.trainer.model(input=image,
                                       gt_loc=gt_loc,
                                       gt_cls=gt_cls,
                                       forward_switch='train',
                                       **train_item_spec)

        if self.trainer.use_amp:
            raise NotImplementedError('Amp is not implemented in algorithm auto lane.')

        #---------------------------------------------------#
        #  loss汇总
        #---------------------------------------------------#
        loss_pos = model_out['loss_pos'].mean()
        loss_neg = model_out['loss_neg'].mean()
        loss_loc = model_out['loss_loc'].mean()

        if self.train_lane:
            loss = loss_loc + (loss_pos + loss_neg)
        else:
            loss = 0

        #---------------------------------------------------#
        #  车道线种类
        #---------------------------------------------------#
        train_lane_type = False
        if "loss_lane_type" in model_out:
            train_lane_type = True
            loss_lane_type = model_out["loss_lane_type"].mean()
            loss+=loss_lane_type
        else:
            loss_lane_type = None

        #---------------------------------------------------#
        #  检测分割
        #---------------------------------------------------#
        train_seg=False
        train_det=False
        loss_det_reg=0.0
        loss_det_cls=0.0
        loss_seg=0.0
        if "loss_det_cls" in model_out:
            train_det=True
            loss_det_cls = model_out["loss_det_cls"].mean()
            loss_det_reg = model_out["loss_det_reg"].mean()
            loss+=(loss_det_cls  + loss_det_reg)
        if "loss_seg" in model_out:
            train_seg=True
            loss_seg = model_out["loss_seg"].mean()
            loss+=loss_seg

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.trainer.model.parameters(), 100)

        self.trainer.optimizer.step()

        if train_det and train_seg:
            dict_summary= {'loss': loss.item(),
                    'cls_pos_loss': loss_pos.item(),
                    'cls_neg_loss': loss_neg.item(),
                    'loc_loss': loss_loc.item(),
                    "seg_loss": loss_seg.item(),
                    "det_loss_cls":loss_det_cls.item(),
                    "det_loss_reg":loss_det_reg.item(),
                    'train_batch_output': None}

        elif not train_det and train_seg:
            dict_summary= {'loss': loss.item(),
                    'cls_pos_loss': loss_pos.item(),
                    'cls_neg_loss': loss_neg.item(),
                    'loc_loss': loss_loc.item(),
                    "seg_loss": loss_seg.item(),
                    'train_batch_output': None}

        elif train_det and not train_seg:
            dict_summary= {'loss': loss.item(),
                    'cls_pos_loss': loss_pos.item(),
                    'cls_neg_loss': loss_neg.item(),
                    'loc_loss': loss_loc.item(),
                    "det_loss_cls":loss_det_cls.item(),
                    "det_loss_reg":loss_det_reg.item(),
                    'train_batch_output': None}

        else:
            dict_summary= {'loss': loss.item(),
                    'cls_pos_loss': loss_pos.item(),
                    'cls_neg_loss': loss_neg.item(),
                    'loc_loss': loss_loc.item(),
                    'train_batch_output': None}

        if train_lane_type:
            dict_summary["lane_type_loss"] = loss_lane_type.item()

        return dict_summary

    # def before_valid(self, logs=None):
    #     """Be called before a batch validation."""
    #     epochs = self.params['epochs']

    def valid_step(self, batch, iou_metric=None):
        """Be called on each batch validing."""
        self.trainer.model.eval()

        image, valid_item_spec = batch

        pointlane_decode = self.trainer.train_loader.dataset.codec_obj
        predict_result = self.trainer.model(input=image, forward_switch='valid')

        # 加入车道线
        predict_result_rearrange=dict()
        predict_result_rearrange["image"]=predict_result["image"].detach().contiguous().cpu().numpy()
        predict_result_rearrange["regression"]=predict_result["regression"].detach().cpu().numpy()
        predict_result_rearrange["classfication"]=predict_result["classfication"].detach().cpu().numpy()

        # 判断是进行分割和检测
        TRAIN_SEG = False
        TRAIN_DET = False

        #---------------------------------------------------#
        #  检测相关
        #---------------------------------------------------#
        if "cls_score" in predict_result:
            predict_result_rearrange["cls_score"]=predict_result["cls_score"].detach().cpu().numpy()
            predict_result_rearrange["bbox_pred"]=predict_result["bbox_pred"].detach().cpu().numpy()
            TRAIN_DET = True
            box_coder = BoxCoder()
            anchor_generator = Anchors(ratios=np.array([0.5, 1, 2]), )
        else:
            box_coder=None
            anchor_generator=None

        #---------------------------------------------------#
        #  分割相关
        #---------------------------------------------------#
        if "output_seg" in predict_result:
            predict_result_rearrange["output_seg"]=predict_result["output_seg"].detach().cpu().numpy()
            TRAIN_SEG = True

        #---------------------------------------------------#
        #  车道线类型检测
        #---------------------------------------------------#

        bundle_result = ChainMap(valid_item_spec, predict_result)
        results = []
        for index, retail_dict_spec in enumerate(multidict_split(bundle_result)):

            #---------------------------------------------------#
            #  1.车道线metric计算
            #---------------------------------------------------#
            if "lane_type" in retail_dict_spec:
                lane_set = pointlane_decode.decode_lane(
                    predict_type=retail_dict_spec['classfication'],
                    predict_loc=retail_dict_spec['regression'],
                    lane_class_preds=retail_dict_spec["lane_type"],
                )
            else:
                lane_set = pointlane_decode.decode_lane(
                    predict_type=retail_dict_spec['classfication'],
                    predict_loc=retail_dict_spec['regression'],
                    lane_class_preds=None,
                )

            lane_nms_set = nms_with_pos(lane_set, thresh=70)
            net_input_image_shape = ujson.loads(retail_dict_spec['net_input_image_shape'])
            src_image_shape = ujson.loads(retail_dict_spec['src_image_shape'])
            lane_order_set = order_lane_x_axis(lane_nms_set, net_input_image_shape['height'])
            scale_x = src_image_shape['width'] / net_input_image_shape['width']
            scale_y = src_image_shape['height'] / net_input_image_shape['height']

            predict_json = convert_lane_to_dict(lane_order_set, scale_x, scale_y)
            target_json = ujson.loads(retail_dict_spec['annot'])
            results.append(dict(pr_result={**predict_json, **dict(Shape=src_image_shape)},
                                gt_result={**target_json, **dict(Shape=src_image_shape)}))

            #---------------------------------------------------#
            #  2.语义分割metric计算
            #---------------------------------------------------#
            if TRAIN_SEG:
                gt_seg = retail_dict_spec["gt_seg"].unsqueeze(0)
                predict_seg = retail_dict_spec["output_seg"].unsqueeze(0).detach().cpu()
                seg_prediction = torch.argmax(predict_seg, dim=1)
                iou_metric.update(seg_prediction, gt_seg)

            #---------------------------------------------------#
            #  3.目标检测metric计算
            #---------------------------------------------------#
            if TRAIN_DET:
                # 原始图
                origin_image = retail_dict_spec["image"].unsqueeze(0).permute(0, 3, 1, 2)

                # 真实标签列表
                annot_path = str(retail_dict_spec["annotation_path"])

                # 预测框
                predict_bbox = retail_dict_spec["bbox_pred"].unsqueeze(0)

                # 预测类别
                predict_score = retail_dict_spec["cls_score"].unsqueeze(0)

                # Anchor
                original_anchors = anchor_generator(origin_image)   # (bs, num_all_achors, 5)

                tmp_name = os.path.basename(annot_path).replace(".json",".txt")
                tmp_dir = os.path.dirname(annot_path).replace("labels_lane","labels_object_tmp_pred")
                im_scales = np.array([1.0/scale_x, 1.0/scale_y, 1.0/scale_x, 1.0/scale_y])

                scores, classes, boxes = det_decoder(box_coder,origin_image, original_anchors, predict_score, predict_bbox, test_conf=0.5)

                scores = scores.data.cpu().numpy()
                classes = classes.data.cpu().numpy()
                boxes = boxes.data.cpu().numpy()
                boxes[:, :4] = boxes[:, :4] / im_scales
                if boxes.shape[1] > 5:
                    boxes[:, 5:9] = boxes[:, 5:9] / im_scales
                scores = np.reshape(scores, (-1, 1))
                classes = np.reshape(classes, (-1, 1))
                cls_dets = np.concatenate([classes, scores, boxes], axis=1)
                keep = np.where(classes > 0)[0]
                out_eval_ = cls_dets[keep, :]
                out_file = os.path.join(tmp_dir, tmp_name)
                with codecs.open(out_file, 'w', 'utf-8') as f:
                    if out_eval_.shape[0] == 0:
                        f.close()
                        continue
                    res = sort_corners(rbox_2_quad(out_eval_[:, 2:]))
                    for k in range(out_eval_.shape[0]):
                        f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                            DET_CLASS_LIST[int(out_eval_[k, 0])],
                            out_eval_[k, 1],
                            res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                            res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                        )

        return {'valid_batch_output': results}

def multidict_split(bundle_dict):
    """Split multi dict to retail dict.

    :param bundle_dict: a buddle of dict
    :type bundle_dict: a dict of list
    :return: retails of dict
    :rtype: list
    """
    retails_list = [dict(zip(bundle_dict, i)) for i in zip(*bundle_dict.values())]
    return retails_list


#---------------------------------------------------#
#  检测validation
#---------------------------------------------------#
def det_decoder(box_coder, ims, anchors, cls_score, bbox_pred, thresh=0.6, nms_thresh=0.2, test_conf=None):
    if test_conf is not None:
        thresh = test_conf
    bboxes = box_coder.decode(anchors, bbox_pred, mode='xywht')
    bboxes = clip_boxes(bboxes, ims)
    scores = torch.max(cls_score, dim=2, keepdim=True)[0]
    keep = (scores >= thresh)[0, :, 0]

    if keep.sum() == 0:
        return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
    scores = scores[:, keep, :]
    anchors = anchors[:, keep, :]
    cls_score = cls_score[:, keep, :]
    bboxes = bboxes[:, keep, :]

    # NMS
    anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
    nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
    output_boxes = torch.cat([
        bboxes[0, anchors_nms_idx, :],
        anchors[0, anchors_nms_idx, :]],
        dim=1
    )

    return [nms_scores, nms_class, output_boxes]

def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted
