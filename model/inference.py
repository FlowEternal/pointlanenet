
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.onnx

# 导入检测相关包
import vega
from vega.common import argment_parser
from vega.datasets.common.utils.auto_lane_codec_utils import nms_with_pos, order_lane_x_axis
from vega.datasets.common.utils.auto_lane_codec_utils import convert_lane_to_dict
from vega.datasets.common.utils.auto_lane_pointlane_codec import PointLaneCodec

import time
#---------------------------------------------------#
#  语义分割
#---------------------------------------------------#
# 语义分割可视化颜色定义

seg_class_color_id = {0:(0,0,0),
                      1:(0,255,0),
                      2:(0,0,255),
                      3:(255,0,0),
                      4:(255,0,255),
                      5:(0,255,255),
                      6:(128,255,255),
                      7:(255,128,0),
                      # 8:(128,0,255)
                   }



#---------------------------------------------------#
#  目标检测
#---------------------------------------------------#
from vega.networks.pytorch.detectors.det_box_coder import BoxCoder
from vega.networks.pytorch.detectors.nms_wrapper import nms
from vega.networks.pytorch.detectors.det_bbox import clip_boxes
from vega.datasets.common.utils.det_bbox import rbox_2_quad
from vega.networks.pytorch.detectors.det.anchors import Anchors


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

#---------------------------------------------------#
#  其它函数
#---------------------------------------------------#
def imagenet_normalize( img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img

def _load_data(args):
    """Load data from path."""
    if not os.path.exists(args.data_path):
        raise("data path is empty, path={}".format(args.data_path))
    else:
        _path = os.path.abspath(args.data_path)
        _files = [(os.path.join(_path, _file)) for _file in os.listdir(_path)]
        _files = [_file for _file in _files if os.path.isfile(_file)]
        return _files

def _load_image_lane(image_file):
    """Load every image."""
    im_tmp = cv2.imread(image_file)

    img = cv2.cvtColor(im_tmp,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    img = imagenet_normalize(img)
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0)
    return img

def _load_image_lane_video(img):
    """Load every image."""
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img.astype(np.float32)
    img = imagenet_normalize(img)
    img = np.expand_dims(np.transpose(img, (2,0,1)),axis=0)
    return img


def _to_tensor(data):
    """Change data to tensor."""
    data = torch.tensor(data)
    if args.device == "GPU":
        return data.cuda()
    else:
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

def _infer(loader, model=None):
    """Choose backend."""
    if vega.is_torch_backend():
        return _infer_pytorch_lane(model, loader, video_test=TEST_VIDEO)

def _infer_pytorch_lane(model, loader, video_test):
    """Infer with pytorch."""
    with torch.no_grad():
        counter=0
        while True:


            #====================================================================#
            # 视频测试
            if video_test:
                _, im_vis = vid.read()
                if im_vis is None:
                    break
                print("process frame %i" % counter)
                # 前处理
                data = _to_tensor(_load_image_lane_video(im_vis)).float()

            # 图片列表测试
            else:
                if counter >=len(loader):
                    break
                img_path = loader[counter]

                print("process image %s" %img_path)
                im_vis = cv2.imread(img_path)

                # 前处理
                data = _to_tensor(_load_image_lane(img_path)).float()
            #====================================================================#
            counter += 1
            # if counter < 2000:
            #     continue

            # 网络前向
            tic = time.time()
            output_dict = model(data,forward_switch="valid")

            if DEPLOY and not DO_LANE_TYPE_DETECT:
                model.eval()
                batch_size = 1
                x = torch.randn(batch_size, 3, 288, 512, requires_grad=True).cuda(0)
                torch.onnx.export(
                                    model,
                                    (x,"deploy"),
                                    "lane_reg.onnx",
                                    export_params=True,
                                    input_names=["input","mode"],
                                    output_names=["regression","classification"],
                                    opset_version=11,
                                    verbose=False,
                                    )
                exit()

            if DEPLOY and DO_LANE_TYPE_DETECT:

                data = data.repeat(BATCH_SIZE,1,1,1)

                batch_size = 1

                dynamic_axes = {
                                'input': {0: 'batch_size'},  # variable lenght axes
                                'regression': {0: 'batch_size'},
                                "classification":{0:"batch_size"},
                                "lane_type":{0:"batch_size"}
                                }


                model.eval()
                torch.onnx.export(
                    model,
                    (data, "deploy"),
                    "lane_reg_type.onnx",
                    export_params=True,
                    input_names=["input", "mode"],
                    output_names=["regression", "classification","lane_type"],
                    opset_version=11,
                    verbose=False,
                    # dynamic_axes=dynamic_axes
                )
                exit()

            #---------------------------------------------------#
            #  语义分割处理后处理
            #---------------------------------------------------#
            if DO_SEGMENTATION:
                segment = output_dict["output_seg"]
                seg_prediction = torch.argmax(segment, dim=1)[0].detach().cpu().numpy()

                # vis
                vis_seg = np.zeros([seg_prediction.shape[0], seg_prediction.shape[1], 3], dtype=np.uint8)
                for cls_id, color in seg_class_color_id.items():
                    vis_seg[seg_prediction == cls_id] = color
                vis_seg = cv2.resize(vis_seg, ORG_SIZE,cv2.INTER_NEAREST)
                im_vis = cv2.addWeighted(im_vis, 0.8, vis_seg, 0.5, 0.0)

                #---------------------------------------------------#
                #  车道线后处理
                #---------------------------------------------------#
            if DO_LANE_DETECT:
                regression=output_dict['regression'][0]
                classfication=F.softmax(output_dict['classfication'][0], -1)
                if DO_LANE_TYPE_DETECT:
                    lane_type_preds = output_dict["lane_type"][0]
                else:
                    lane_type_preds = None
                lane_set = pointlane.decode_lane(predict_type=classfication,
                                                 predict_loc=regression,
                                                 exist_threshold=CONF_THRESHOLD,
                                                 lane_class_preds=lane_type_preds,
                                                 )

                lane_nms_set = nms_with_pos(lane_set,
                                            thresh=NMS_LINE_THRESHOLD,
                                            use_mean_dist=USE_MEAN)

                print("total process time is %.2f" % (1000*(time.time() - tic)))

                lane_order_set = order_lane_x_axis( list(lane_nms_set), NET_INPUT_IMAGE_SHAPE['height'])
                scale_x = SRC_IMAGE_SHAPE['width'] / TARGET_SIZE[0]
                scale_y = SRC_IMAGE_SHAPE['height'] / TARGET_SIZE[1]

                predict_json = convert_lane_to_dict( lane_order_set , scale_x, scale_y)["Lines"]

                for tmp_line in predict_json:
                    score = tmp_line["score"]
                    pts = tmp_line["points"]
                    if DO_LANE_TYPE_DETECT:
                        lane_type = tmp_line["lane_type"]
                    else:
                        lane_type = "Lane"
                    length = len(pts)
                    if length < 3:
                        continue
                    for idx,pt in enumerate(pts):
                        point = ( int(pt["x"]) , int(pt["y"]) )

                        if idx + 1 == length:
                            break
                        else:
                            points_after = (int(pts[idx + 1]["x"]), int(pts[idx + 1]["y"]))

                        im_vis = cv2.line(im_vis,
                                       tuple(point),
                                       tuple(points_after),
                                       color=(255,255,0),
                                       thickness=10)

                    # 置信度文本显示
                    pt_ = pts[2]
                    pt_txt = [ int(pt_["x"]) , int(pt_["y"]) ]
                    if pt_txt[0] <0:
                        pt_txt[0] = 30

                    if pt_txt[0] > SRC_IMAGE_SHAPE['width']:
                        pt_txt[0] = SRC_IMAGE_SHAPE['width'] - 700

                    cv2.putText(im_vis, "%s: %.2f" % (lane_type,float(score) ),
                                (pt_txt[0],pt_txt[1]-50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0, (0, 255, 255), 7)


            #---------------------------------------------------#
            #  目标检测处理后处理
            #---------------------------------------------------#
            if DO_DETECTION:
                im_scale_x = float(TARGET_SIZE[0]) / float(ORG_SIZE[0])
                im_scale_y = float(TARGET_SIZE[1]) / float(ORG_SIZE[1])
                im_scales = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])

                # 获取结果
                cls_score, bbox_pred = output_dict["cls_score"], output_dict["bbox_pred"]
                original_anchors = anchor_generator(data)  # (bs, num_all_achors, 5)

                scores, classes, boxes = det_decoder(box_coder,
                                                     data,
                                                     original_anchors,
                                                     cls_score,
                                                     bbox_pred,
                                                     test_conf=TEST_CONFIG)

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

                res = sort_corners(rbox_2_quad(out_eval_[:, 2:]))
                for k in range(out_eval_.shape[0]):
                    det_info = '{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                        DET_CLASS_LIST[int(out_eval_[k, 0])],
                        out_eval_[k, 1],
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                        res[k, 4], res[k, 5], res[k, 6], res[k, 7])

                    det_info_vis = det_info.split(" ")
                    det_cls_name = det_info_vis[0]
                    det_cls_conf = float(det_info_vis[1])
                    det_info = det_cls_name + " %.2f" % det_cls_conf

                    pt1 = (int(res[k, 0]), int(res[k, 1]))
                    pt2 = (int(res[k, 2]), int(res[k, 3]))
                    pt3 = (int(res[k, 4]), int(res[k, 5]))
                    pt4 = (int(res[k, 6]), int(res[k, 7]))
                    cv2.line(im_vis, pt1, pt2, DET_COLOR, 4)
                    cv2.line(im_vis, pt2, pt3, DET_COLOR, 4)
                    cv2.line(im_vis, pt3, pt4, DET_COLOR, 4)
                    cv2.line(im_vis, pt4, pt1, DET_COLOR, 4)

                    bbox = out_eval_[k, 2:]
                    t_size = cv2.getTextSize(det_info, font, fontScale=fontScale, thickness=thickness)[0]
                    c1 = tuple(bbox[:2].astype('int'))
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                    cv2.rectangle(im_vis, c1, c2, DET_COLOR, -1)  # filled
                    cv2.putText(im_vis, det_info, (c1[0], c1[1] - 5), font, fontScale, [0, 0, 0],
                                thickness=thickness,
                                lineType=cv2.LINE_AA)

            #---------------------------------------------------#
            #  保存显示
            #---------------------------------------------------#
            if video_test:
                if DISPLAY:
                    cv2.imshow('vis', im_vis)
                    cv2.waitKey(1)
                video_writer.write(im_vis)

                # if counter > 500:
                #     break

            else:
                cv2.imwrite(img_path.replace("images","images_vis"),im_vis)

def parse_args_parser():
    """Parse parameters."""
    parser = argment_parser('Vega Inference.')
    parser.add_argument("-c", "--model_desc",
                        default="models/desc_25.json",
                        type=str,
                        help="model description file, generally in json format, contains 'module' node.")

    parser.add_argument("-m", "--model",
                        default="models/model_25.pth",
                        type=str,
                        help="model weight file, usually ends with pth, ckpl, etc.")

    parser.add_argument("-df", "--data_format",
                        default="detection",
                        type=str,
                        choices=["classification", "c",
                                 "super_resolution", "s",
                                 "segmentation", "g",
                                 "detection", "d"],
                        help="data type, "
                        "classification: some pictures file in a folder, "
                        "super_resolution: some low resolution picture in a folder, "
                        "segmentation: , "
                        "detection: . "
                        "'classification' is default" )

    parser.add_argument("-dp",
                        "--data_path",
                        default="demo/images",
                        type=str,
                        help="the folder where the file to be inferred is located.")

    parser.add_argument("-b",
                        "--backend",
                        default="pytorch",
                        type=str,
                        choices=["pytorch", "tensorflow", "mindspore"],
                        help="set training platform")

    parser.add_argument("-d", "--device", default="GPU", type=str,
                        choices=["CPU", "GPU", "NPU"],
                        help="set training device")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # python -m onnxsim input_onnx_model output_onnx_model
    DEPLOY = False
    BATCH_SIZE = 1

    TEST_VIDEO = False
    DISPLAY = False
    ORG_SIZE = (1920, 1080)
    # ORG_SIZE = (2560,1440)
    TARGET_SIZE = (512, 288)

    #---------------------------------------------------#
    #  多任务模型配置
    #---------------------------------------------------#
    # DO_LANE_DETECT = True
    # DO_LANE_TYPE_DETECT = False
    # DO_SEGMENTATION = True
    # DO_DETECTION = True
    # CONF_THRESHOLD = 0.5
    # USE_MEAN = False
    # NMS_LINE_THRESHOLD = 100
    # MODEL_PATH = "models/trained_mt/mt_resnet34_epoch100.pth"
    # MODEL_CONFIG = "models/archs/mt.json"

    #---------------------------------------------------#
    #  车道线模型配置 -- 只检测车道线位置
    #---------------------------------------------------#
    # DO_LANE_DETECT = True
    # DO_LANE_TYPE_DETECT = False
    # DO_SEGMENTATION = False
    # DO_DETECTION = False
    # CONF_THRESHOLD = 0.4
    # USE_MEAN = False
    # NMS_LINE_THRESHOLD = 100

    # MODEL_PATH = "models/trained_lane/lane_resnet34_reg_epoch50.pth"
    # MODEL_CONFIG = "models/archs/lane_resnet34_reg.json"
    # MODEL_PATH = "tasks/test.pth"
    # MODEL_CONFIG = "models/archs/lane_nas_reg.json"

    #---------------------------------------------------#
    #  车道线模型配置 -- 车道线类型位置
    #---------------------------------------------------#
    DO_LANE_DETECT = True
    DO_LANE_TYPE_DETECT = True
    DO_SEGMENTATION = False
    DO_DETECTION = False
    CONF_THRESHOLD = 0.5
    USE_MEAN = False
    NMS_LINE_THRESHOLD = 100
    MODEL_PATH = "models/trained_lane/lane_resnet34_cls_epoch100_cfg1.pth"
    # MODEL_PATH = "models/trained_lane/lane_resnet34_cls_epoch100_cfg2.pth"
    # MODEL_PATH = "models/trained_lane/epoch_4.pth"
    MODEL_CONFIG = "models/archs/lane_resnet34_cls.json"

    #---------------------------------------------------#
    #  超参数设定
    #---------------------------------------------------#
    # 车道线
    SCALE_INVARIANCE = True
    ANCHOR_PER_LINE = 72

    # 检测
    anchor_generator = Anchors(ratios=np.array([0.5, 1, 2]),)
    box_coder = BoxCoder()
    TEST_CONFIG = 0.35
    fontScale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    DET_COLOR = (0, 255, 0)
    NET_INPUT_IMAGE_SHAPE = {"width": TARGET_SIZE[0], "height": TARGET_SIZE[1], "channel": 3}
    SRC_IMAGE_SHAPE = {"width": ORG_SIZE[0], "height": ORG_SIZE[1], "channel": 3}

    # 加载视频
    if DISPLAY:
        cv2.namedWindow("vis",cv2.WINDOW_FREERATIO)

    # vid = cv2.VideoCapture("demo/video/test_video_one.avi")
    # video_output = "demo/video_vis/test_video_one_vis.avi"

    vid = cv2.VideoCapture("demo/video/test_video_two.mp4")
    video_output = "demo/video_vis/test_video_two_vis.mp4"

    # vid = cv2.VideoCapture("demo/video/test_video_three.avi") # 最难
    # video_output = "demo/video_vis/test_video_three_vis.avi"

    # vid = cv2.VideoCapture("demo/video/test_left_video.avi")
    # video_output = "demo/video_vis/test_left_video_vis.avi"

    # vid = cv2.VideoCapture("demo/video/test_right_video.avi")
    # video_output = "demo/video_vis/test_right_video_vis.avi"

    # vid = cv2.VideoCapture("demo/video/LEFT_BACK.avi")
    # video_output = "demo/video_vis/left_back_vis.avi"

    # vid = cv2.VideoCapture("demo/video/LEFT_FRONT.avi")
    # video_output = "demo/video_vis/left_front_vis.avi"

    # vid = cv2.VideoCapture("demo/video/RIGHT_FRONT.avi")
    # video_output = "demo/video_vis/right_front_vis.avi"

    # vid = cv2.VideoCapture("demo/video/RIGHT_BACK.avi")
    # video_output = "demo/video_vis/right_back_vis.avi"

    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(video_output, codec, 10, ORG_SIZE)

    # 可视化images
    output_dir_img = "demo/images_vis"
    if not os.path.exists(output_dir_img): os.makedirs(output_dir_img)

    # 可视化video
    output_dir_video = "demo/video_vis"
    if not os.path.exists(output_dir_video): os.makedirs(output_dir_video)

    #---------------------------------------------------#
    #  定义车道线解码器
    #---------------------------------------------------#
    pointlane = PointLaneCodec(
                               input_width=TARGET_SIZE[0],
                               input_height=TARGET_SIZE[1],
                               anchor_stride=16,
                               points_per_line=ANCHOR_PER_LINE,
                               class_num=2,
                               scale_invariance=SCALE_INVARIANCE,
                               with_lane_cls=DO_LANE_TYPE_DETECT,
                               lane_cls_num=4,
                               )

    args = parse_args_parser()
    args.model = MODEL_PATH
    args.model_desc = MODEL_CONFIG
    vega.set_backend(args.backend, args.device)

    print("Start building model.")
    model = _get_model(args)

    print("Start loading data.")
    loader = _load_data(args)

    print("Start inferencing.")
    _infer(loader, model)

