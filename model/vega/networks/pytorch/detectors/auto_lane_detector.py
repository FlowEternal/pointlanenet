# -*- coding: utf-8 -*-
"""Defined faster rcnn detector."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vega.common import ClassType, ClassFactory
from vega.modules.module import Module
from vega.datasets.common.utils.auto_lane_pointlane_codec import PointLaneCodec

#---------------------------------------------------#
#  检测相关函数
#---------------------------------------------------#
from vega.networks.pytorch.detectors.detector import DetDecoder
from vega.networks.pytorch.detectors.det.anchors import Anchors
from vega.networks.pytorch.detectors.det.losses_det import IntegratedLoss
from vega.networks.pytorch.detectors.det_box_coder import BoxCoder

#---------------------------------------------------#
#  分割相关函数
#---------------------------------------------------#
from vega.networks.pytorch.detectors.seg import SegDecoder
from vega.networks.pytorch.detectors.loss import CrossEntropyLoss

#---------------------------------------------------#
#  骨干网络
#---------------------------------------------------#
from vega.networks.pytorch.detectors.backbone import resnet34                   # resnet
from vega.networks.pytorch.detectors.yolov5.yolo_backbone import YOLOBACKBONE   # yolo

def get_img_whc(img):
    """Get image whc by src image.

    :param img: image to transform.
    :type: ndarray
    :return: image info
    :rtype: dict
    """
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        c = 1
    elif len(img_shape) == 3:
        h, w, c = img_shape
    else:
        raise NotImplementedError()
    return dict(width=w, height=h, channel=c)

def multidict_split(bundle_dict):
    """Split multi dict to retail dict.

    :param bundle_dict: a buddle of dict
    :type bundle_dict: a dict of list
    :return: retails of dict
    :rtype: list
    """
    retails_list = [dict(zip(bundle_dict, i)) for i in zip(*bundle_dict.values())]
    return retails_list

def find_k_th_small_in_a_tensor(target_tensor, k_th):
    """Like name, this function will return the k the of the tensor."""
    val, idxes = torch.topk(target_tensor, k=k_th, largest=False)
    return val[-1]

def huber_fun(x):
    """Implement of hunber function."""
    absx = torch.abs(x)
    r = torch.where(absx < 1, x * x / 2, absx - 0.5)
    return r

@ClassFactory.register(ClassType.NETWORK)
class AutoLaneDetector(Module):
    """Faster RCNN."""

    def __init__(self, desc):
        """Init faster rcnn.

        :param desc: config dict
        """
        super(AutoLaneDetector, self).__init__()
        self.desc = desc

        #---------------------------------------------------#
        #  构建骨干网络
        #---------------------------------------------------#
        self.RESNET = desc["backbone_resnet"]           # 是否采用resnet34作为主干网络
        self.CSPNET = desc["backbone_yolo"]             # 是否采用yolov5主干网络作为主干网络

        def build_module(net_type_name):
            return ClassFactory.get_cls(ClassType.NETWORK, desc[net_type_name].type)(desc[net_type_name])

        if self.RESNET:
            model_path = "models/archs/resnet34.pth"
            self.backbone = resnet34(pretrained=True, model_path=model_path)

        elif self.CSPNET:
            model_config = "models/archs/yolov5s.yaml"
            self.backbone = YOLOBACKBONE(model_config)

        else:
            self.backbone = build_module('backbone')

        # 通道数信息
        if self.RESNET or self.CSPNET:
            desc["neck"]["in_channels"] = [64, 128, 256, 512]
            desc["head"]["base_channel"] = 512
            desc["head_seg"]["out_channel"] = [32, 64, 128, 256]

        #---------------------------------------------------#
        #  构建分支头
        #---------------------------------------------------#
        # 权重策略
        self.INIT_LANE = desc["init_lane"]              # 权重初始化策略
        self.DYNAMIC_WEIGHT = desc["dynamic_weight"]    # 动态loss权重调整策略是否开启
        self.ALPHA = 10
        self.NEGATIVE_RATIO = 15
        self.SCALE_INVARIANCE = desc["head"]["scale_invariance"]

        #---------------------------------------------------#
        #  车道线分支头
        #---------------------------------------------------#
        # 维度相关信息
        self.num_class = desc["num_class"]
        self.stride = desc["head"]["input_size"]["anchor_stride"]
        self.input_width = desc["head"]["input_size"]["width"]
        self.input_height = desc["head"]["input_size"]["height"]
        self.interval = desc["head"]["input_size"]["interval"]
        self.feat_width = int(self.input_width / self.stride)
        self.feat_height = int(self.input_height / self.stride)
        self.points_per_line = int(self.input_height / self.interval)

        # 车道线损失函数权重
        self.lane_weight_cls = desc["head"]["weight_cls"]
        self.lane_weight_reg = desc["head"]["weight_reg"]
        self.lane_cls_focal_loss = desc["head"]["use_focal"]

        # 车道线类别损失
        self.focal_loss_alpha = 1.0
        self.focal_loss_gamma = 2.0
        self.do_lane_classification = desc["head"]["do_classify"]
        self.classify_weight=desc["head"]["weight_classify"]
        self.lane_class_name_list = desc["head"]["lane_class_list"]
        self.lane_class_num = len(self.lane_class_name_list)
        self.lane_class_weight = desc["head"]["lane_class_weight_positive"]

        # 车道线编码/解码函数
        self.pointlane_codec = PointLaneCodec(input_width=self.input_width,
                                              input_height=self.input_height,
                                              anchor_stride=self.stride,
                                              points_per_line=self.points_per_line,
                                              class_num=self.num_class,
                                              scale_invariance=self.SCALE_INVARIANCE,
                                              with_lane_cls=self.do_lane_classification,
                                              lane_cls_num=self.lane_class_num
                                              )

        self.neck = build_module('neck')
        self.head = build_module('head')
        self.LANE_POINTS_NUM_DOWN = self.points_per_line + 1
        self.LANE_POINTS_NUM_UP = self.points_per_line + 1

        if self.DYNAMIC_WEIGHT:
            self.lane_weight_dy = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        if self.INIT_LANE:
            self.init_module(self.backbone)
            self.init_module(self.head)
            self.init_module(self.neck)

        #---------------------------------------------------#
        #  语义分割分支
        #---------------------------------------------------#
        self.do_segmentation = desc["head_seg"]["do_seg"]
        if self.do_segmentation:
            # segmentation decoder part
            if self.RESNET or self.CSPNET:
                self.channel_dimension_seg_encode = [64, 128, 256, 512]
            else:
                self.channel_dimension_seg_encode = desc["backbone"]["out_channels"]
            self.channel_dimension_seg_decode = desc["head_seg"]["out_channels"]
            self.segment_class_list = desc["head_seg"]["seg_class_list"]
            self.segment_weight = desc["head_seg"]["weight"]

            if self.DYNAMIC_WEIGHT:
                self.segment_weight_dy = nn.Parameter(torch.tensor(0.0), requires_grad=True)

            self.segdecoder = SegDecoder(num_ch_enc=self.channel_dimension_seg_encode,
                                         num_ch_dec=self.channel_dimension_seg_decode,
                                         num_output_channels=len(self.segment_class_list))

            # 损失函数
            weight = torch.ones(len(self.segment_class_list))
            weight[0] = 0.1 # background
            self.loss_fn_seg = CrossEntropyLoss(
                class_weights=weight,
                use_top_k=False,
                top_k_ratio=0.3,
                future_discount=1.0,
                use_focal=True)

        #---------------------------------------------------#
        # 目标检测分支
        #---------------------------------------------------#
        self.do_detection = desc["head_detect"]["do_detection"]
        if self.do_detection:
            # detection decoder part
            if self.RESNET or self.CSPNET:
                self.channel_dimension = [64, 128, 256, 512]
            else:
                self.channel_dimension = desc["backbone"]["out_channels"]

            self.detect_class_list = desc["head_detect"]["detect_class_list"]
            self.det_weight_cls = desc["head_detect"]["weight_cls"]
            self.det_weight_reg = desc["head_detect"]["weight_reg"]

            if self.DYNAMIC_WEIGHT:
                self.detect_weight_dy = nn.Parameter(torch.tensor(0.0), requires_grad=True) # 以reg为基准

            self.detectdecoder = DetDecoder(fpn_in_channels=self.channel_dimension,
                                            class_num=len(self.detect_class_list))
            self.anchor_generator = Anchors(ratios=np.array([0.5, 1, 2]),)
            self.loss_det = IntegratedLoss(func='smooth')

            self.box_coder = BoxCoder()
            self.nt = 0.0

            # 检测不需要按照init_module初始化 注意 这里的坑

        #---------------------------------------------------#
        #  记录epoch
        #---------------------------------------------------#
        self.total_epoch = int(desc["epoch"])
        self.iter_per_epoch = int(desc["iter_epoch"])
        self.iter_counter = 0

    @staticmethod
    def init_module(input_module):
        for m in input_module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def calc_mining_param(process,alpha):
        if process < 0.1:
            bf_weight = 1.0
        elif  process > 0.3:
            bf_weight = alpha
        else:
            bf_weight = 5*(alpha-1)*process+1.5-0.5*alpha
        return bf_weight

    def extract_feat(self, img):
        """Public compute of input.

        :param img: input image
        :return: feature map after backbone and neck compute
        """
        x = self.backbone(img)
        x = self.neck(x[0:4])
        return x

    def extract_feat_mtl(self, img):
        """Public compute of input.

        :param img: input image
        :return: feature map after backbone and neck compute
        """
        x = self.backbone(img)
        return x

    def extract_feat_mtl_resnet(self, img):
        """Public compute of input.

        :param img: input image
        :return: feature map after backbone and neck compute
        """
        x = self.backbone(img, output_feat = True)
        return x

    def forward(self, input, forward_switch='calc', **kwargs):
        """Call default forward function."""
        if forward_switch == 'train':
            return self.forward_train(input, **kwargs)
        elif forward_switch == 'valid':
            return self.forward_valid(input)
        elif forward_switch == 'calc':
            return self.forward_calc_params_and_flops(input, **kwargs)
        elif forward_switch == 'deploy':
            return self.forward_deploy(input)

    def forward_calc_params_and_flops(self, input, **kwargs):
        """Just for calc paramters."""
        feat = self.extract_feat(input)
        predict = self.head(feat)
        return predict

    def cal_loss_cls(self,cls_targets,cls_preds, use_focal_loss = True):
        #---------------------------------------------------#
        #  计算mask
        #---------------------------------------------------#
        cls_targets = cls_targets[..., 1].view(-1)
        pmask = cls_targets > 0
        nmask = ~ pmask
        fpmask = pmask.float()
        fnmask = nmask.float()

        #---------------------------------------------------#
        #  处理
        #---------------------------------------------------#
        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])
        total_postive_num = torch.sum(fpmask)
        total_negative_num = torch.sum(fnmask)  # Number of negative entries to select
        negative_num = torch.clamp(total_postive_num * self.NEGATIVE_RATIO, max=total_negative_num, min=1).int()
        positive_num = torch.clamp(total_postive_num, min=1).int()

        if use_focal_loss:

            #---------------------------------------------------#
            #  focal loss
            #---------------------------------------------------#
            bg_fg_predict = F.softmax(cls_preds, dim=-1)
            weight_focal = torch.pow(-bg_fg_predict + 1., self.focal_loss_gamma)
            focal_item = -self.ALPHA * weight_focal * torch.log(bg_fg_predict+1e-8)
            total_focal_pos = focal_item[pmask,1].mean()
            total_focal_neg = focal_item[nmask,0].mean()

            return total_focal_pos,total_focal_neg,pmask,positive_num

        else:
            #---------------------------------------------------#
            #  类别损失
            #---------------------------------------------------#
            bg_fg_predict = F.log_softmax(cls_preds, dim=-1)
            fg_predict = bg_fg_predict[..., 1]
            bg_predict = bg_fg_predict[..., 0]
            max_hard_pred = find_k_th_small_in_a_tensor(bg_predict[nmask].detach(), negative_num)
            fnmask_ohem = (bg_predict <= max_hard_pred).float() * nmask.float()
            total_cross_pos = -torch.sum(self.ALPHA * fg_predict * fpmask)
            total_cross_neg = -torch.sum(self.ALPHA * bg_predict * fnmask_ohem)

            #---------------------------------------------------#
            #  取平均
            #---------------------------------------------------#
            total_cross_pos = total_cross_pos / positive_num
            total_cross_neg = total_cross_neg / positive_num

            return total_cross_pos, total_cross_neg, pmask, positive_num

    def cal_loss_regress(self,pmask,positive_num,loc_targets,loc_preds,is_horizon=True):
        #---------------------------------------------------#
        #  计算回归损失
        #---------------------------------------------------#
        loc_preds = loc_preds.view(-1, loc_preds.shape[-1])
        loc_targets = loc_targets.view(-1, loc_targets.shape[-1])
        length_weighted_mask = torch.ones_like(loc_targets)

        if is_horizon:
            length_weighted_mask[..., self.points_per_line_horizon] = self.ALPHA
            length_weighted_mask[..., self.points_per_line_horizon + 1] = self.ALPHA
        else:
            length_weighted_mask[..., self.points_per_line + 1] = self.ALPHA
            length_weighted_mask[..., self.points_per_line] = self.ALPHA

        valid_lines_mask = pmask.unsqueeze(-1).expand_as(loc_targets)
        valid_points_mask = (loc_targets != 0)
        unified_mask = length_weighted_mask.float() * valid_lines_mask.float() * valid_points_mask.float()
        smooth_huber = huber_fun(loc_preds - loc_targets) * unified_mask
        loc_smooth_l1_loss = torch.sum(smooth_huber, -1)
        point_num_per_gt_anchor = torch.sum(valid_points_mask.float(), -1).clamp(min=1)
        total_loc = torch.sum(loc_smooth_l1_loss / point_num_per_gt_anchor)

        total_loc = total_loc / positive_num

        return total_loc

    def cal_focal_lane_type(self,pred_sigmoid, target, alpha_positive=0.25):
        # https://zhuanlan.zhihu.com/p/80594704
        # focal loss损失函数 + GHM损失函数
        # 这里计算sigmoid函数的2分类focal loss

        # target is one
        weight_one = torch.pow(-pred_sigmoid + 1., self.focal_loss_gamma)
        focal_one = -alpha_positive * weight_one * torch.log(pred_sigmoid+1e-8)
        loss_one = target * focal_one

        # target is zero
        weight_zero = torch.pow(pred_sigmoid , self.focal_loss_gamma)
        focal_zero = - (1 - alpha_positive) * weight_zero * torch.log(1 - pred_sigmoid+1e-8)
        loss_zero = (1-target)* focal_zero

        # return (loss_one + loss_zero).mean()
        return (loss_one + loss_zero).mean()

    def cal_lanetype_loss(self, class_targets, lane_type_target, lane_target_predicts):
        # TODO
        cls_targets = class_targets[..., 1].view(-1)
        pmask = cls_targets > 0

        lane_type_target = lane_type_target.view(-1,self.lane_class_num)
        lane_target_predicts = lane_target_predicts.view(-1,self.lane_class_num)

        lane_type_gt_positive = lane_type_target[pmask]
        lane_type_predicts_positive = lane_target_predicts[pmask]

        focal_lane_type=0.0
        # for normal and abnormal line
        lane_pred_first = lane_type_predicts_positive[:,0]
        lane_gt_first = lane_type_gt_positive[:,0]
        focal_lane_type+=self.cal_focal_lane_type(lane_pred_first,lane_gt_first,self.lane_class_weight[0])

        # for the rest lane label
        mask_positive = (lane_gt_first==0)
        normal_pred = lane_type_predicts_positive[mask_positive]
        normal_gt = lane_type_gt_positive[mask_positive]

        for i in range(1,self.lane_class_num):
            focal_lane_type+=self.cal_focal_lane_type(normal_pred[:,i],normal_gt[:,i],self.lane_class_weight[i])

        return focal_lane_type

    def cal_lanetype_loss_softmax(self, class_targets, lane_type_target, lane_target_predicts):
        cls_targets = class_targets[..., 1].view(-1)
        pmask = cls_targets > 0

        lane_type_target = lane_type_target.view(-1,self.lane_class_num)
        lane_target_predicts = lane_target_predicts.view(-1,self.lane_class_num)

        lane_type_gt_positive = lane_type_target[pmask]
        lane_type_predicts_positive = F.softmax(lane_target_predicts[pmask],dim=1)

        # focal loss for multi class classification
        weight = torch.pow(-lane_type_predicts_positive + 1., self.focal_loss_gamma)
        weight_set = torch.tensor(self.lane_class_weight).unsqueeze(0).repeat(lane_type_predicts_positive.shape[0],1).cuda()

        focal = - self.focal_loss_alpha * weight * torch.log(lane_type_predicts_positive + 1e-10) * weight_set
        focal_lane_type = torch.mean(lane_type_gt_positive * focal)

        return focal_lane_type

    @staticmethod
    def dynamic_weight(loss_term, weight_term, dynamic_weight_term, divider):
        dynamic_weight_term = dynamic_weight_term.to(loss_term.device)
        return loss_term * weight_term * ( 1 / (torch.exp(dynamic_weight_term))) + dynamic_weight_term/divider,

    def forward_train(self, input, **kwargs):
        """Forward compute between train process.

        :param input: input data
        :return: losses
        :rtype: torch.tensor
        """
        image = input
        self.iter_counter+=1

        if self.RESNET:
            feat_raw = self.extract_feat_mtl_resnet(image)[1:]
        elif self.CSPNET:
            feat_raw = self.extract_feat_mtl_resnet(image)
        else:
            feat_raw = self.extract_feat_mtl(image)

        #---------------------------------------------------#
        #  车道线前向 + 损失计算
        #---------------------------------------------------#
        neck_lane = self.neck(feat_raw[0:4])
        predict = self.head(neck_lane)
        loc_targets = kwargs['gt_loc']
        cls_targets = kwargs['gt_cls']
        loc_preds = predict['predict_loc']
        cls_preds = predict['predict_cls']

        total_cross_pos, total_cross_neg, pmask, positive_num = self.cal_loss_cls(cls_targets,
                                                                                  cls_preds,self.lane_cls_focal_loss)

        total_loc = self.cal_loss_regress(pmask, positive_num, loc_targets, loc_preds, is_horizon=False)

        if self.DYNAMIC_WEIGHT:

            loss_dict = dict(
                    loss_pos=total_cross_pos * self.lane_weight_cls*
                             ( 1 / (torch.exp(self.lane_weight_dy.to(total_cross_pos.device))) ) +
                             self.lane_weight_dy.to(total_cross_pos.device)/3.0,
                    loss_neg=total_cross_neg* self.lane_weight_cls*
                             ( 1 / (torch.exp(self.lane_weight_dy.to(total_cross_neg.device))) ) +
                             self.lane_weight_dy.to(total_cross_neg.device)/3.0,
                    loss_loc=total_loc* self.lane_weight_reg *
                             ( 1 / (torch.exp(self.lane_weight_dy.to(total_loc.device))) ) +
                             self.lane_weight_dy.to(total_loc.device)/3.0
                )

        else:
            loss_dict = dict(
                    loss_pos=total_cross_pos * self.lane_weight_cls,
                    loss_neg=total_cross_neg* self.lane_weight_cls,
                    loss_loc=total_loc* self.lane_weight_reg
                )

        #---------------------------------------------------#
        #  车道线分类损失
        #---------------------------------------------------#
        if self.do_lane_classification:
            lane_type_predicts = predict["predict_lane_type"]
            lane_type_labels = kwargs["gt_lane_type"].cuda().float()
            total_lane_type = self.cal_lanetype_loss_softmax(cls_targets, lane_type_labels, lane_type_predicts)

            if self.DYNAMIC_WEIGHT:
                loss_dict["loss_lane_type"] = total_lane_type * self.classify_weight * \
                                              (1 / (torch.exp(self.lane_weight_dy.to(total_lane_type.device)))) \
                                              + self.lane_weight_dy.to(total_lane_type.device)

            else:

                loss_dict["loss_lane_type"] = total_lane_type * self.classify_weight

        if not self.do_segmentation and self.do_detection:
            # 只做检测
            # ---------------------------------------------------#
            #  目标检测分支
            # ---------------------------------------------------#
            cls_score, bbox_pred = self.detectdecoder(feat_raw)
            det_label = kwargs['gt_det'].cuda().float()

            anchors_list, offsets_list, cls_list, var_list = [], [], [], []
            original_anchors = self.anchor_generator(image)  # (bs, num_all_achors, 5)
            anchors_list.append(original_anchors)
            bboxes = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()

            ratio = int(self.iter_counter / float(self.iter_per_epoch))  # 判断在哪个epoch
            bf_weight = self.calc_mining_param(ratio / float(self.total_epoch), 0.3)

            loss_det_cls, loss_det_reg = self.loss_det(cls_score,
                                                   bbox_pred,
                                                   anchors_list[-1],
                                                   bboxes,
                                                   det_label,
                                                   md_thres=0.6,
                                                   mining_param=(bf_weight, 1 - bf_weight, 5))
            if self.DYNAMIC_WEIGHT:
                loss_dict["loss_det_cls"] = loss_det_cls * self.det_weight_cls* \
                                            ( 1 / (torch.exp(self.detect_weight_dy.to(loss_det_cls.device))) ) + \
                                            self.detect_weight_dy.to(loss_det_cls.device)/2.0

                loss_dict["loss_det_reg"] = loss_det_reg * self.det_weight_reg* \
                                            ( 1 / (torch.exp(self.detect_weight_dy.to(loss_det_reg.device))) ) + \
                                            self.detect_weight_dy.to(loss_det_reg.device)/2.0
            else:
                loss_dict["loss_det_cls"] = loss_det_cls * self.det_weight_cls
                loss_dict["loss_det_reg"] = loss_det_reg * self.det_weight_reg

        if self.do_segmentation and not self.do_detection:
            # 只做分割
            #---------------------------------------------------#
            #  语义分割分支
            #---------------------------------------------------#
            output_seg, _ = self.segdecoder(feat_raw)
            loss_seg = self.loss_fn_seg(output_seg, kwargs['gt_seg'].cuda().long())
            if self.DYNAMIC_WEIGHT:
                loss_dict["loss_seg"] = loss_seg * self.segment_weight* \
                                        ( 1 / (torch.exp(self.segment_weight_dy.to(loss_seg.device))) ) +\
                                        self.segment_weight_dy.to(loss_seg.device)
            else:
                loss_dict["loss_seg"] = loss_seg * self.segment_weight

        if self.do_detection and self.do_segmentation:
            # 分割检测都做
            #---------------------------------------------------#
            #  目标检测分支
            #---------------------------------------------------#
            cls_score, bbox_pred = self.detectdecoder(feat_raw)
            det_label = kwargs['gt_det'].cuda().float()
            original_anchors = self.anchor_generator(image)   # (bs, num_all_achors, 5)
            bboxes = self.box_coder.decode(original_anchors, bbox_pred, mode='xywht').detach()
            ratio = int(self.iter_counter/float(self.iter_per_epoch) ) # 判断在哪个epoch
            bf_weight = self.calc_mining_param( ratio/ float(self.total_epoch), 0.3)

            loss_det_cls, loss_det_reg = self.loss_det(cls_score,
                                                   bbox_pred,
                                                   original_anchors,
                                                   bboxes,
                                                   det_label,
                                                   md_thres=0.6,
                                                   mining_param=(bf_weight, 1-bf_weight, 5))


            if self.DYNAMIC_WEIGHT:
                loss_dict["loss_det_cls"] = loss_det_cls * self.det_weight_cls* \
                                            ( 1 / (torch.exp(self.detect_weight_dy.to(loss_det_cls.device))) ) + \
                                            self.detect_weight_dy.to(loss_det_cls.device)/2.0

                loss_dict["loss_det_reg"] = loss_det_reg * self.det_weight_reg* \
                                            ( 1 / (torch.exp(self.detect_weight_dy.to(loss_det_reg.device))) ) + \
                                            self.detect_weight_dy.to(loss_det_reg.device)/2.0

            else:
                loss_dict["loss_det_cls"] = loss_det_cls * self.det_weight_cls
                loss_dict["loss_det_reg"] = loss_det_reg * self.det_weight_reg

            #---------------------------------------------------#
            #  语义分割分支
            #---------------------------------------------------#
            output_seg, _ = self.segdecoder(feat_raw)
            loss_seg = self.loss_fn_seg(output_seg, kwargs['gt_seg'].cuda(non_blocking=True).long())

            if self.DYNAMIC_WEIGHT:
                loss_dict["loss_seg"] = loss_seg * self.segment_weight* \
                                        ( 1 / (torch.exp(self.segment_weight_dy.to(loss_seg.device))) ) + \
                                        self.segment_weight_dy.to(loss_seg.device)
            else:
                loss_dict["loss_seg"] = loss_seg * self.segment_weight

        return loss_dict

    def forward_valid(self, input):
        """Forward compute between inference.

        :param input: input data must be image
        :return: groundtruth result and predict result
        :rtype: dict
        """
        image = input

        if self.RESNET:
            feat_raw = self.extract_feat_mtl_resnet(image)[1:]
        elif self.CSPNET:
            feat_raw = self.extract_feat_mtl_resnet(image)
        else:
            feat_raw = self.extract_feat_mtl(image)

        neck_lane = self.neck(feat_raw[0:4])
        predict = self.head(neck_lane)

        if self.do_lane_classification:
            predict_result = dict(
                image=image.permute((0, 2, 3, 1)).contiguous(),
                regression=predict['predict_loc'],
                classfication=F.softmax(predict['predict_cls'], -1),
                lane_type=predict['predict_lane_type'],
            )

        else:

            predict_result = dict(
                image=image.permute((0, 2, 3, 1)).contiguous(),
                regression=predict['predict_loc'],
                classfication=F.softmax(predict['predict_cls'], -1),
            )

        if not self.do_segmentation and self.do_detection:
            # 只做检测
            # ---------------------------------------------------#
            #  目标检测分支
            # ---------------------------------------------------#
            cls_score, bbox_pred = self.detectdecoder(feat_raw)
            predict_result["cls_score"] = cls_score
            predict_result["bbox_pred"] = bbox_pred

        if self.do_segmentation and not self.do_detection:
            # 只做分割
            # ---------------------------------------------------#
            #  语义分割分支
            # ---------------------------------------------------#
            output_seg, _ = self.segdecoder(feat_raw)
            predict_result["output_seg"] = output_seg

        if self.do_detection and self.do_segmentation:
            # 分割检测都做
            # ---------------------------------------------------#
            #  目标检测分支
            # ---------------------------------------------------#
            cls_score, bbox_pred = self.detectdecoder(feat_raw)

            # ---------------------------------------------------#
            #  语义分割分支
            # ---------------------------------------------------#
            output_seg, _ = self.segdecoder(feat_raw)
            predict_result["cls_score"] = cls_score
            predict_result["bbox_pred"] = bbox_pred
            predict_result["output_seg"] = output_seg

        # 直接返回预测结果
        return predict_result

    def forward_deploy(self, input):
        """Forward compute between inference.

        :param input: input data must be image
        :return: groundtruth result and predict result
        :rtype: dict
        """
        image = input

        if self.RESNET:
            feat_raw = self.extract_feat_mtl_resnet(image)[1:]
        elif self.CSPNET:
            feat_raw = self.extract_feat_mtl_resnet(image)
        else:
            feat_raw = self.extract_feat_mtl(image)

        neck_lane = self.neck(feat_raw[0:4])
        predict = self.head(neck_lane)

        regression = predict['predict_loc'],
        classfication = F.softmax(predict['predict_cls'], -1),

        if self.do_lane_classification:
            lane_type = predict['predict_lane_type']
            return regression, classfication,lane_type
        else:
            return regression,classfication
