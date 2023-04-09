# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for CurveLane dataset."""

import os
import json
import shutil
import cv2
import numpy as np
import torch.utils.data.dataloader

from more_itertools import grouper
from vega.common import FileOps
from vega.common import ClassFactory, ClassType
from vega.datasets.common.dataset import Dataset

#---------------------------------------------------#
#  车道线相关包
#---------------------------------------------------#
from vega.datasets.conf.auto_lane import AutoLaneConfig
from vega.datasets.common.utils.auto_lane_pointlane_codec import PointLaneCodec
from vega.datasets.common.utils.auto_lane_utils import get_img_whc, imread, create_train_subset, create_test_subset
from vega.datasets.common.utils.auto_lane_utils import load_lines, resize_by_wh, bgr2rgb, imagenet_normalize, load_json

#---------------------------------------------------#
#  数据增强函数
#---------------------------------------------------#
import imgaug as ia
import imgaug.augmenters as iaa

# 车道线
from imgaug.augmentables.lines import LineStringsOnImage
from imgaug.augmentables.lines import LineString as ia_LineString

# 目标检测
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.polys import Polygon as ia_Polygon
from vega.datasets.common.utils.det_bbox import quad_2_rbox, mask_valid_boxes

#  语义分割增强函数
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

#---------------------------------------------------#
#  目标检测类别
#---------------------------------------------------#
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

#---------------------------------------------------#
#  语义分割类别定义
#---------------------------------------------------#
seg_class_list = {"background":0,
                  "pedestrian_area":1,
                  "self_area":2,
                  "obstacle_area":3,
                  "road_area":4,
                  "marking_area":5,
                  "vehicle_area":6,
                  "marking_general_area":7,
                  "marking_pavement_area":8
                  }

seg_class_list_key = seg_class_list.keys()

# 语义分割可视化颜色定义
seg_class_color = {"background":(0,0,0),
                   "pedestrian_area":(0,255,0),
                  "self_area":(0,0,255),
                  "obstacle_area":(255,0,0),
                  "road_area":(255,0,255),
                  "marking_area":(0,255,255),
                  "vehicle_area":(128,255,255),
                  "marking_general_area":(255,128,0),
                  "marking_pavement_area": (128,0,255)
                   }

# 语义分割可视化颜色定义
seg_class_color_id = {0:(0,0,0),
                      1:(0,255,0),
                      2:(0,0,255),
                      3:(255,0,0),
                      4:(255,0,255),
                      5:(0,255,255),
                      6:(128,255,255),
                      7:(255,128,0),
                      8:(128,0,255)
                   }

# 处理polyfit RANK warning
import warnings

def _culane_line_to_curvelane_dict(culane_lines):
    curvelane_lines = []
    for culane_line_spec in culane_lines:
        curvelane_lien_spec = [{'x': x, 'y': y} for x, y in grouper(map(float, culane_line_spec.split(' ')), 2)]
        curvelane_lines.append(curvelane_lien_spec)
    return dict(Lines=curvelane_lines)

def _lane_argue(*,
                image,
                lane_src,
                do_flip,
                det_label=None,
                seg_label=None,
                with_type=False,
                do_split=False,
                split_ratio=None
                ):
    #---------------------------------------------------#
    #  定义增强序列
    #---------------------------------------------------#
    color_shift = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 1.5)),
        iaa.LinearContrast((1.5, 1.5), per_channel=False),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(0, iaa.Multiply((0.7, 1.3)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(1, iaa.Multiply((0.1, 2)))),
        iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV, from_colorspace=iaa.CSPACE_RGB,
                           children=iaa.WithChannels(2, iaa.Multiply((0.5, 1.5)))),
    ])

    geometry_trans_list = [
            iaa.Fliplr(),
            iaa.TranslateX(px=(-16, 16)),
            iaa.ShearX(shear=(-15, 15)),
            iaa.Rotate(rotate=(-15, 15))
        ]

    if do_flip:
        geometry_trans_list.append(iaa.Flipud())

    if do_split:
        # top right down left
        split_one = iaa.Crop(percent=([0, 0.2], [1 - split_ratio], [0, 0], [0, 0.15]), keep_size=True)# 右边是1-ratio
        split_two = iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [split_ratio]), keep_size=True)
        split_shift = iaa.OneOf([split_one, split_two])

    else:
        geometry_trans_list.append(iaa.Crop(percent=([0, 0.2], [0, 0.15], [0, 0], [0, 0.15]), keep_size=True))
        split_shift = None

    posion_shift = iaa.SomeOf(4, geometry_trans_list)

    if do_split:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=1.0, then_list=split_shift), # 以0.5概率去split debug时候1.0
            iaa.Sometimes(p=0.6, then_list=posion_shift)
        ], random_order=True)

    else:
        aug = iaa.Sequential([
            iaa.Sometimes(p=0.6, then_list=color_shift),
            iaa.Sometimes(p=0.6, then_list=posion_shift)
        ], random_order=True)

    # 开始数据增强
    lines_tuple = [[(float(pt['x']), float(pt['y'])) for pt in line_spec] for line_spec in lane_src['Lines']]
    lss = [ia_LineString(line_tuple_spec) for line_tuple_spec in lines_tuple]
    lsoi = LineStringsOnImage(lss, shape=image.shape)
    args = {"images":[image],"line_strings":[lsoi]}

    # 做语义分割增强
    if seg_label is not None:
        segmap = SegmentationMapsOnImage( seg_label , shape=image.shape)
        args.update({"segmentation_maps":[segmap]})

    # 做目标检测增强
    if det_label is not None:
        polygon_list = [ia_Polygon(one_det_poly.reshape(4,2)) for one_det_poly in det_label]
        deoi = PolygonsOnImage(polygon_list, shape=image.shape)
        args.update({"polygons": [deoi]})

    batch = ia.Batch(**args)
    batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
    image_aug = batch_aug.images_aug[0]

    # 增强line
    # lsoi_aug = batch_aug.line_strings_aug[0].clip_out_of_image() # 这个很重要
    lsoi_aug = batch_aug.line_strings_aug[0] # 这里不clip_out_of_image()

    lane_aug = [[dict(x= float(int(kpt.x)), y=float(int(kpt.y))) for kpt in shapely_line.to_keypoints()]
                for shapely_line in lsoi_aug]

    # 这里数据增强后clip_out_of_image有可能出现某条车道线被完全clip out
    # 这样和标签数量不匹配
    # 要把相应标签也一起过滤掉
    if with_type:
        len_org = len(lane_src['Lines'])
        type_list = [item for item in lane_src["Labels"] ]

        # 过滤超出图像边界的线：
        lane_aug_org =  [[dict(x= float(int(kpt.x)), y=float(int(kpt.y))) for kpt in shapely_line.to_keypoints()]
            for shapely_line in batch_aug.line_strings_aug[0]]

        mask = np.ones(len_org, dtype=np.bool)
        for idx, one_line in enumerate(lane_aug_org):
            counter = 0
            for idx_inner,one_pt in enumerate(one_line):
                one_pt_tmp = (float(one_pt['x']), float(one_pt['y']))
                if not ((0 <= one_pt_tmp[0] <= image.shape[1]) and (0 <= one_pt_tmp[1] <= image.shape[0])):
                    counter+=1

            if counter == len(one_line):
                mask[idx] = False # 如果所有线都超出了图像，那么直接mask掉
            else:
                mask[idx] = True

        type_list = list(np.array(type_list)[mask])
        lane_aug = list(np.array(lane_aug)[mask])
        aug_result = {"images": image_aug, "lane_aug": dict(Lines=lane_aug, Labels=type_list)}
        assert len(lane_aug) == len(type_list)

    else:
        aug_result = {"images": image_aug, "lane_aug": dict(Lines=lane_aug,Labels=None)}

    # 增强detection
    if det_label is not None:
        deoi_aug = batch_aug.polygons_aug[0] # 这里clip out of image 会有问题，所以不clip
        det_label_aug = np.vstack([np.hstack([ np.array([int(kpt.x),int(kpt.y)]) for kpt in det_polygon.to_keypoints()]) for det_polygon in deoi_aug])
        aug_result.update({"det_aug":det_label_aug})

    # 增强分割掩码
    if seg_label is not None:
        segmap_aug = batch_aug.segmentation_maps_aug[0]
        aug_result.update({"seg_aug":segmap_aug})

    return aug_result

def _read_curvelane_type_annot(annot_path):
    return load_json(annot_path)

def _read_culane_type_annot(annot_path):
    return _culane_line_to_curvelane_dict(load_lines(annot_path))

@ClassFactory.register(ClassType.DATASET)
class AutoLaneDataset(Dataset):
    """This is the class of CurveLane dataset, which is a subclass of Dataset.

    :param train: `train`, `val` or `test`
    :type train: str
    :param cfg: config of this datatset class
    :type cfg: yml file that in the entry
    """

    config = AutoLaneConfig()

    def __init__(self, **kwargs):
        """Construct the dataset."""
        super().__init__(**kwargs)

        self.args.data_path = FileOps.download_dataset(self.args.data_path)

        #---------------------------------------------------#
        #  参数设定
        #---------------------------------------------------#
        self.TRAIN_LANE_OWN_DATA = self.args.train_own
        self.TRAIN_LANE_WITH_TYPE = self.args.train_lane_with_type
        self.LANE_CLS_TYPE = self.args.lane_cls_num
        self.DO_SPLIT_IMAGE = self.args.do_split
        self.LIST_NAME = self.args.list_name

        # 检测相关
        self.TRAIN_DETECT = self.args.train_detect
        self.det_class_list = DET_CLASS_LIST
        self.det_num_classes = len(self.det_class_list)
        self.det_class_to_ind = dict(zip(self.det_class_list, range(self.det_num_classes)))

        # 分割相关
        self.TRAIN_SEG = self.args.train_seg

        # 准备数据集
        dataset_pairs = dict(
            train=create_train_subset(self.args.data_path,is_own=self.TRAIN_LANE_OWN_DATA,with_seg=self.TRAIN_SEG,with_detect=self.TRAIN_DETECT,list_type=self.LIST_NAME),
            test=create_test_subset(self.args.data_path,is_own=self.TRAIN_LANE_OWN_DATA,with_seg=self.TRAIN_SEG,with_detect=self.TRAIN_DETECT,list_type=self.LIST_NAME),
            val=create_test_subset(self.args.data_path,is_own=self.TRAIN_LANE_OWN_DATA,with_seg=self.TRAIN_SEG,with_detect=self.TRAIN_DETECT,list_type=self.LIST_NAME)
        )

        if self.mode not in dataset_pairs.keys():
            raise NotImplementedError(f'mode should be one of {dataset_pairs.keys()}')
        self.image_annot_path_pairs = dataset_pairs.get(self.mode)
        self.points_per_line = int(self.args.network_input_height / self.args.interval)
        self.points_per_line_horizon = int(self.args.network_input_width / self.args.interval)

        # 是否插值 如果插值 即为沿长到版边 和原代码就完全一致
        self.interpolate = self.args.use_interpolation
        self.codec_obj = PointLaneCodec(input_width=self.args.network_input_width,
                                        input_height=self.args.network_input_height,
                                        anchor_stride=self.args.anchor_stride,
                                        points_per_line=self.points_per_line,
                                        class_num=self.args.num_class,
                                        do_interpolate=self.interpolate,
                                        with_lane_cls = self.TRAIN_LANE_WITH_TYPE,
                                        lane_cls_num=self.LANE_CLS_TYPE,
                                        )

        self.encode_lane = self.codec_obj.encode_lane
        self.stride = self.args.anchor_stride

        read_funcs = dict(
            CULane=_read_culane_type_annot,
            CurveLane=_read_curvelane_type_annot,
        )

        self.args.dataset_format = "CurveLane"
        if self.args.dataset_format not in read_funcs:
            raise NotImplementedError(f'dataset_format should be one of {read_funcs.keys()}')
        self.read_annot = read_funcs.get(self.args.dataset_format)
        self.with_aug = self.args.get('with_aug', False)

        # collate function
        if self.mode=="train":
            self.is_valid = False
        else:
            self.is_valid = True

        #---------------------------------------------------#
        #  检测需要移动真值valid标签
        #---------------------------------------------------#
        if self.TRAIN_DETECT and self.is_valid:
            # 上次预测标签清理
            target_dir_pred = os.path.join(self.args.data_path,"labels_object_tmp_pred")
            if not os.path.exists(target_dir_pred):
                os.makedirs(target_dir_pred)
            else:
                list_all = os.listdir(target_dir_pred)
                for tmp_name in list_all:
                    os.remove(os.path.join(target_dir_pred, tmp_name))

            # 临时真值标签文件
            target_dir = os.path.join(self.args.data_path,"labels_object_tmp")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            else:
                list_all = os.listdir(target_dir)
                for tmp_name in list_all:
                    os.remove(os.path.join(target_dir, tmp_name))

            # 然后移动真值标签
            for tmp_pair in self.image_annot_path_pairs:
                tmp_det = tmp_pair["annot_path_detect"]
                tmp_det_target = tmp_det.replace("labels_object","labels_object_tmp")
                shutil.copy(tmp_det,tmp_det_target)


        self.collate_fn = Collater(target_height=self.args.network_input_height,
                                   target_width=self.args.network_input_width,
                                   is_seg=self.TRAIN_SEG,
                                   is_det=self.TRAIN_DETECT,
                                   is_valid=self.is_valid,
                                   with_line_type=self.TRAIN_LANE_WITH_TYPE)

    def __len__(self):
        """Get the length.

        :return: the length of the returned iterators.
        :rtype: int
        """
        return len(self.image_annot_path_pairs)

    def __getitem__(self, idx):
        """Get an item of the dataset according to the index.

        :param idx: index
        :type idx: int
        :return: an item of the dataset according to the index
        :rtype: dict
        """
        if self.mode == 'train':
            return self.prepare_train_img(idx)
        elif self.mode == 'val':
            return self.prepare_test_img(idx)
        elif self.mode == 'test':
            return self.prepare_test_img(idx)
        else:
            raise NotImplementedError


    def prepare_train_img(self, idx):
        """Prepare an image for training.

        :param idx:index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]

        #---------------------------------------------------#
        #  加入车道线
        #---------------------------------------------------#
        lane_object = self.read_annot(target_pair['annot_path'])
        image_arr = imread(target_pair['image_path'])
        whc = get_img_whc(image_arr)

        if self.TRAIN_LANE_OWN_DATA:
            lane_object = self.parse_own_label(lane_object)

        # ---------------------------------------------------#
        #  加入语义分割
        # ---------------------------------------------------#
        if self.TRAIN_SEG:
            seg_arr = cv2.imread(target_pair['annot_path_seg'], cv2.IMREAD_UNCHANGED)
        else:
            seg_arr = None

        # ---------------------------------------------------#
        #  加入目标检测
        # ---------------------------------------------------#
        if self.TRAIN_DETECT:
            roidb = self.det_load_annotation(target_pair['annot_path_detect'])
            gt_inds = np.where(roidb['gt_classes'] != 0)[0]
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds]
            gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        else:
            classes = None
            bboxes = None
            gt_boxes = None

        if DEBUG:
            self.draw_label_on_image(image_arr, lane_object, bboxes, classes, seg_arr, "img_org.png")

        if self.with_aug:
            try:
                if self.DO_SPLIT_IMAGE:
                    do_split_possible, split_ratio = self.cal_split(image_arr,lane_object)
                    do_split = do_split_possible
                else:
                    do_split = False
                    split_ratio = None

                aug_dict = _lane_argue(image=image_arr,
                                       lane_src=lane_object,
                                       do_flip=self.args.do_flip,
                                       det_label=bboxes,
                                       seg_label=seg_arr,
                                       with_type=self.TRAIN_LANE_WITH_TYPE,
                                       do_split=do_split,
                                       split_ratio=split_ratio
                                       )

                # 先看检测 如果没有目标了直接不增强了
                if self.TRAIN_DETECT:
                    bboxes = aug_dict["det_aug"]
                    # 去除超出边界的检测框 TODO
                    mask_ = self.clip_out_bboxes(image_arr, bboxes)
                    # 关键语句
                    assert np.array(mask_).sum()!=0
                    gt_boxes = gt_boxes[mask_]
                    classes = classes[mask_]
                    bboxes = bboxes[mask_]

                    # 进一步处理检测标签
                    mask = mask_valid_boxes(quad_2_rbox(bboxes, 'xywha'), return_mask=True)
                    bboxes = bboxes[mask]
                    gt_boxes = gt_boxes[mask]
                    classes = classes[mask]
                    for i, bbox in enumerate(bboxes):
                        gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode='xyxya')
                        gt_boxes[i, 5] = classes[i] # 最后的包围盒标签 原图尺度

                # 覆盖分割
                if self.TRAIN_SEG:
                    seg_arr = aug_dict["seg_aug"].arr[:, :, 0].astype(np.uint8)  # 分割标签 原图尺度

                # 覆盖原图和车道线
                image_arr = aug_dict["images"]
                lane_object = aug_dict["lane_aug"]

                if DEBUG:
                    self.draw_label_on_image(image_arr, lane_object,bboxes, classes,seg_arr, "img_aug.png")

                encode_type, encode_loc,encode_cls = self.encode_lane(lane_object=lane_object,
                                                           org_width=whc['width'],
                                                           org_height=whc['height'])

            except AssertionError:
                if self.TRAIN_DETECT:
                    roidb = self.det_load_annotation(target_pair['annot_path_detect'])
                    gt_inds = np.where(roidb['gt_classes'] != 0)[0]
                    bboxes = roidb['boxes'][gt_inds, :]
                    classes = roidb['gt_classes'][gt_inds]
                    gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
                    for i, bbox in enumerate(bboxes):
                        gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode='xyxya')
                        gt_boxes[i, 5] = classes[i]  # 最后的包围盒标签 原图尺度
                else:
                    gt_boxes = None

                if DEBUG:
                    self.draw_label_on_image(image_arr, lane_object, bboxes, classes,seg_arr, "img_aug.png")

                encode_type, encode_loc, encode_cls = self.encode_lane(lane_object=lane_object,
                                                           org_width=whc['width'],
                                                           org_height=whc['height'])

        else:
            if self.TRAIN_DETECT:
                for i, bbox in enumerate(bboxes):
                    gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode='xyxya')
                    gt_boxes[i, 5] = classes[i]  # 最后的包围盒标签 原图尺度

            encode_type, encode_loc,encode_cls = self.encode_lane(lane_object=lane_object,
                                                       org_width=whc['width'],
                                                       org_height=whc['height'])

        network_input_image = bgr2rgb(resize_by_wh(img=image_arr,
                                                   width= self.args.network_input_width,
                                                   height=self.args.network_input_height))


        if self.args.scale_invariance:
            # up anchor
            encode_loc[:, self.points_per_line + 2: 2 * self.points_per_line + 2] /= self.args.interval
            # down anchor
            encode_loc[:, :self.points_per_line] /= self.args.interval

        item = dict(
            net_input_image=imagenet_normalize(img=network_input_image),
            net_input_image_mode='RGB',
            net_input_image_shape=dict(width=self.args.network_input_width, height=self.args.network_input_height, channel=3),
            src_image_shape=whc,
            src_image_path=target_pair['image_path'],
            annotation_path=target_pair['annot_path'],
            annotation_src_content=lane_object,
            regression_groundtruth=encode_loc,
            classfication_groundtruth=encode_type,
            lane_cls_groundtruth=encode_cls,
            segmentation_groundtruth=seg_arr,
            detection_groundtruth=gt_boxes,
        )

        result = dict(image=np.transpose(item['net_input_image'], (2, 0, 1)).astype('float32'),
                      gt_loc=item['regression_groundtruth'].astype('float32'),
                      gt_cls=item['classfication_groundtruth'].astype('float32'),
                      gt_lane_type=item["lane_cls_groundtruth"],
                      gt_seg=item["segmentation_groundtruth"],
                      gt_det=item["detection_groundtruth"],
                      src_image_shape=whc)
        return result

    def prepare_test_img(self, idx):
        """Prepare an image for testing.

        :param idx: index
        :type idx: int
        :return: an item of data according to the index
        :rtype: dict
        """
        target_pair = self.image_annot_path_pairs[idx]
        image_arr = imread(target_pair['image_path'])
        lane_object = self.read_annot(target_pair['annot_path'])
        whc = get_img_whc(image_arr)

        if self.TRAIN_LANE_OWN_DATA:
            lane_object = self.parse_own_label(lane_object)
        else:
            lane_object["Labels"] = None

        # ---------------------------------------------------#
        #  加入语义分割
        # ---------------------------------------------------#
        if self.TRAIN_SEG:
            seg_arr = cv2.imread(target_pair['annot_path_seg'], cv2.IMREAD_UNCHANGED)
        else:
            seg_arr = None

        # ---------------------------------------------------#
        #  加入目标检测
        # ---------------------------------------------------#
        if self.TRAIN_DETECT:
            roidb = self.det_load_annotation(target_pair['annot_path_detect'])
            gt_inds = np.where(roidb['gt_classes'] != 0)[0]
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds]
            gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
            for i, bbox in enumerate(bboxes):
                gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode='xyxya')
                gt_boxes[i, 5] = classes[i]  # 最后的包围盒标签 原图尺度

        else:
            gt_boxes = None


        network_input_image = bgr2rgb(resize_by_wh(img=image_arr, width=self.args.network_input_width,
                                                   height=self.args.network_input_height))

        item = dict(
            net_input_image=imagenet_normalize(img=network_input_image),
            net_input_image_mode='RGB',
            net_input_image_shape=dict(width=self.args.network_input_width, height=self.args.network_input_height, channel=3),
            src_image_shape=whc,
            src_image_path=target_pair['image_path'],
            annotation_path=target_pair['annot_path'],
            annotation_src_content=lane_object,
            regression_groundtruth=None,
            classfication_groundtruth=None,
            segmentation_groundtruth=seg_arr,
            detection_groundtruth=gt_boxes,

        )

        result = dict(image=np.transpose(item['net_input_image'], (2, 0, 1)).astype('float32'),
                      net_input_image_shape=json.dumps(item['net_input_image_shape']),
                      src_image_shape=json.dumps(item['src_image_shape']),
                      annot=json.dumps(item['annotation_src_content']),
                      src_image_path=item['src_image_path'],
                      annotation_path=item['annotation_path'],
                      gt_seg=item["segmentation_groundtruth"],
                      gt_det=item["detection_groundtruth"],
                      )

        return result

    @staticmethod
    def cal_split(image,lane_object):
        height , width = image.shape[0], image.shape[1]
        k0_list = []
        k1_list = []
        all_lines = []
        for one_lane in lane_object["Lines"]:
            x_list = []
            y_list = []
            one_line_pts = []
            for pt_index in range(len(one_lane)):
                one_pt = (int(float(one_lane[pt_index]["x"])), height - int(float(one_lane[pt_index]["y"])))
                x_list.append(one_pt[0])
                y_list.append(one_pt[1])
                one_line_pts.append(one_pt)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    coeff = np.polyfit(x_list, y_list, 1)
                except np.RankWarning:
                    return False,None
                except:
                    return False, None
            k0 = coeff[1]
            k1 = coeff[0]
            k0_list.append(k0)
            k1_list.append(k1)
            all_lines.append(one_line_pts)

        # 进行逻辑判断
        k1_list = np.array(k1_list)
        sorted_k1 = np.sort(k1_list)
        index = np.argsort(k1_list)
        if np.all(sorted_k1>=0) or np.all(sorted_k1) <=0:
            do_split_possible = False
            split_ratio = None
        else:
            index_left_lane = np.where(sorted_k1 <=0)[0][0] # 负得最大的那个为左
            left_lane_index = index[ index_left_lane ]
            right_lane_index = index[-1] # 正得最大的那个为左

            left_lane_pts = np.array(all_lines[left_lane_index])
            right_lane_pts = np.array(all_lines[right_lane_index])

            left_lane_pts_sort = left_lane_pts[ np.argsort((left_lane_pts)[:,1],axis=0)  ]
            right_lane_pts_sort = right_lane_pts[ np.argsort((right_lane_pts)[:,1],axis=0)  ]

            left_x_ = left_lane_pts_sort[0,0]
            right_x_ = right_lane_pts_sort[0,0]
            do_split_possible = True
            split_ratio = (left_x_ + right_x_) / 2.0 / width

        return do_split_possible, split_ratio

    @staticmethod
    def draw_line_on_image(image, lane_object, save_name):
        im_vis_org = image.copy()
        for one_lane in lane_object["Lines"]:
            rd_color = (int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)),
                        int(np.random.randint(0, 255)))
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)
        cv2.imwrite(save_name, im_vis_org)

    @staticmethod
    def draw_label_on_image(image, lane_object, bboxes, class_list,seg_arr, save_name):
        im_vis_org = image.copy()

        # 语义分割
        seg_arr_vis = np.zeros_like(image)
        for key,value in seg_class_color_id.items():
            seg_arr_vis[seg_arr==key]=value
        im_vis_org = cv2.addWeighted(im_vis_org,0.5,seg_arr_vis,0.5,0.0)

        # 车道线
        for one_lane in lane_object["Lines"]:
            rd_color = (0,255,0)
            for pt_index in range(len(one_lane) - 1):
                one_pt = one_lane[pt_index]
                one_pt_next = one_lane[pt_index + 1]
                one_pt = (int(float(one_pt["x"])), int(float(one_pt["y"])))
                one_pt_ = (int(float(one_pt_next["x"])), int(float(one_pt_next["y"])))
                print(one_pt)
                cv2.line(im_vis_org, one_pt, one_pt_, rd_color, 3)

        # 目标检测
        for idx,one_box in enumerate(bboxes):
            pt1=(one_box[0],one_box[1])
            pt2=(one_box[2],one_box[3])
            pt3=(one_box[4],one_box[5])
            pt4=(one_box[6],one_box[7])
            cv2.line(im_vis_org,pt1,pt2,(0,255,0),2)
            cv2.line(im_vis_org,pt2,pt3,(0,0,255),2)
            cv2.line(im_vis_org,pt3,pt4,(0,0,255),2)
            cv2.line(im_vis_org,pt4,pt1,(0,0,255),2)

            fontScale = 0.5
            thickness = 1
            font = cv2.FONT_HERSHEY_COMPLEX
            pt_txt = (pt1[0], pt1[1] - 5)

            cv2.putText(im_vis_org, str(class_list[idx]), pt_txt, font, fontScale,
                        [0, 0, 0], thickness=thickness,lineType=cv2.LINE_AA)

        cv2.imwrite(save_name, im_vis_org)

    @staticmethod
    def clip_out_bboxes(image, bboxes):
        height, width = image.shape[0], image.shape[1]
        mask = np.ones(bboxes.shape[0],dtype=np.bool)
        pt_flag = [False, False, False, False]
        for idx,one_box in enumerate(bboxes):
            pt1=(one_box[0],one_box[1])
            if  not ((0 <= pt1[0] <= width) and ( 0 <= pt1[1] <= height)):
                pt_flag[0] = True

            pt2=(one_box[2],one_box[3])
            if  not ((0 <= pt2[0] <= width) and ( 0 <= pt2[1] <= height)):
                pt_flag[1] = True

            pt3=(one_box[4],one_box[5])
            if  not ((0 <= pt3[0] <= width) and ( 0 <= pt3[1] <= height)):
                pt_flag[2] = True

            pt4=(one_box[6],one_box[7])
            if  not ((0 <= pt4[0] <= width) and ( 0 <= pt4[1] <= height)):
                pt_flag[3] = True

            if np.array(pt_flag).sum() ==4:
                mask[idx] = False

        return mask

    # ---------------------------------------------------#
    #  ADDED FUNCTION
    # ---------------------------------------------------#
    # function 1
    @staticmethod
    def parse_own_label(labels):
        lane_list = {"Lines": [], "Labels": []}
        for one_line in labels["shapes"]:
            labels = one_line["label"]

            #---------------------------------------------------#
            #  TODO 注意 这里原始标注时候有road edeg 要过滤掉
            #---------------------------------------------------#
            if labels =="line-road-edge":
                continue
            pts = one_line["points"]
            one_line_list = [{"x": pt[0], "y": pt[1]} for pt in pts]
            lane_list["Lines"].append(one_line_list)
            lane_list["Labels"].append(labels)
        assert len(lane_list["Lines"])==len(lane_list["Labels"])
        return lane_list

    # function 2 检测标签读取
    def det_load_annotation(self, filename):
        boxes, gt_classes = [], []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('\n')
            for obj in objects:
                if len(obj) != 0:
                    class_name = obj.split(",")[0]
                    box = obj.split(",")[1:9]
                    label = self.det_class_to_ind[class_name]
                    box = [eval(x) for x in box]
                    boxes.append(box)
                    gt_classes.append(label)
        return {'boxes': np.array(boxes, dtype=np.int32), 'gt_classes': np.array(gt_classes)}

class Collater(object):
    def __init__(self,
                 target_width,
                 target_height,
                 is_det=True,
                 is_seg=True,
                 is_valid=False,
                 with_line_type=False):
        self.target_width = target_width
        self.target_height = target_height
        self.is_det = is_det
        self.is_seg = is_seg
        self.is_valid=is_valid
        self.with_line_type=with_line_type

    def __call__(self, batch):
        image_data = np.stack([item["image"] for item in batch]) # images
        image_data = torch.from_numpy(image_data)

        if self.is_valid:
            gt_loc=None
            gt_cls=None
        else:
            gt_loc = np.stack([item["gt_loc"] for item in batch]) # location
            gt_loc = torch.from_numpy(gt_loc)
            gt_cls = np.stack([item["gt_cls"] for item in batch]) # cls
            gt_cls = torch.from_numpy(gt_cls)

        img_shape_list = np.stack([item["src_image_shape"] for item in batch]) # cls

        #---------------------------------------------------#
        #  处理segmentation
        #---------------------------------------------------#
        if self.is_seg:
            gt_seg = np.stack([cv2.resize(item["gt_seg"],(self.target_width,self.target_height),cv2.INTER_NEAREST)
                               for item in batch]) # seg
            gt_seg = torch.from_numpy(gt_seg)
        else:
            gt_seg=None

        #---------------------------------------------------#
        #  处理包围盒
        #---------------------------------------------------#
        if self.is_det:
            bboxes = [item["gt_det"] for item in batch] # det
            num_params = bboxes[0].shape[-1]
            max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
            padded_boxes = np.ones([image_data.shape[0], max_num_boxes, num_params]) * -1
            for i in range(image_data.shape[0]):
                bbox = bboxes[i]

                if self.is_valid:
                    img_shape_dict = json.loads(img_shape_list[i])
                else:
                    img_shape_dict = img_shape_list[i]
                org_width, org_height = img_shape_dict["width"],img_shape_dict["height"]
                im_scale_x = (self.target_width) / float(org_width)
                im_scale_y = (self.target_height) / float(org_height)
                im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])

                if num_params < 9:
                    bbox[:, :4] = bbox[:, :4] * im_scale
                else:
                    bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))

                padded_boxes[i, :bbox.shape[0], :] = bbox
            padded_boxes = torch.from_numpy(padded_boxes)
        else:
            padded_boxes = None

        if self.is_valid:
            output_dict = dict(image=image_data)
        else:
            output_dict = dict(image=image_data,
                               gt_loc=gt_loc,
                               gt_cls=gt_cls,
                               )

        if not self.is_valid:
            if self.with_line_type:
                output_dict["gt_lane_type"] =torch.from_numpy(np.stack([item["gt_lane_type"] for item in batch]))

        # 如果是valid，接着将annot信息提取出来
        if self.is_valid:
            output_dict["src_image_shape"] = img_shape_list
            output_dict["annotation_path"] = np.stack([item["annotation_path"] for item in batch])
            output_dict["annot"] = np.stack([item["annot"] for item in batch])
            output_dict["net_input_image_shape"] = np.stack([item["net_input_image_shape"] for item in batch])

        if self.is_seg and self.is_det:
            output_dict["gt_seg"]=gt_seg
            output_dict["gt_det"]=padded_boxes
            return output_dict

        elif self.is_seg and not self.is_det:
            output_dict["gt_seg"]=gt_seg
            return output_dict

        elif not self.is_seg and self.is_det:
            output_dict["gt_det"]=padded_boxes
            return output_dict

        else:
            return output_dict

DEBUG = False
if __name__ == '__main__':
    # 输入测试参数
    input_dict = {"mode":"val",
                  "network_input_width":512,
                  "network_input_height":288,
                  "interval":4,
                  "anchor_stride":16,
                  "num_class":2,
                  "scale_invariance": True,
                  "with_aug": True,
                  "do_flip":False,
                  "data_path": "/data/zdx/Data/data_curvelane",
                  "list_name": "list",
                  "train_seg":False,
                  "train_detect": False,
                  "train_own":True,
                  "use_interpolation":True,
                  "train_lane_with_type":True,
                  "lane_cls_num":9,"do_split":False
                  }

    lane_dataset = AutoLaneDataset( **input_dict)

    trainloader = torch.utils.data.dataloader.DataLoader(lane_dataset,
                                                              batch_size=4,
                                                              num_workers=0,
                                                              shuffle=False,
                                                              drop_last=False,
                                                              pin_memory=True,
                                                              collate_fn=lane_dataset.collate_fn,
                                                              )

    one_data = iter(trainloader).__next__()

    for key,value in one_data.items():
        if not isinstance(value,list):
            print(key, value.shape)
        else:
            print(key)
            for elem in value:
                print(elem)