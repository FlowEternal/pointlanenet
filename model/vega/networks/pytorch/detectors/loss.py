# ==================================================================
# Author    : Dongxu Zhan
# Time      : 2021/7/27 14:01
# File      : loss.py
# Function  : loss function collection
# ==================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utility.utility import

#---------------------------------------------------#
#  segmentation loss
#---------------------------------------------------#
class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights,
                 ignore_index=255,
                 use_top_k=False,
                 top_k_ratio=1.0,
                 future_discount=1.0,
                 use_focal = True,
                 gamma = 2.0,alpha = 1.0
                 ):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.use_focal = use_focal

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, prediction, target):
        b, c, h, w = prediction.shape

        if self.use_focal:
            eps: float = 1e-8
            input_soft: torch.Tensor = F.softmax(prediction, dim=1) + eps

            # create the labels one hot tensor
            one_hot = torch.zeros_like(prediction,dtype=target.dtype).cuda()

            target_one_hot = one_hot.scatter_(1, target.unsqueeze(1), 1.0) + eps

            # print(target_one_hot.shape)

            weight = torch.pow(-input_soft + 1., self.gamma)
            weight_set = self.class_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,weight.shape[2],weight.shape[3])
            focal = -self.alpha * weight * torch.log(input_soft) * weight_set.to(weight.device)
            loss = torch.sum(target_one_hot * focal.to(target_one_hot.device) , dim=1)
            loss = loss.view(b,-1)

        else:
            loss = F.cross_entropy(
                prediction,
                target,
                ignore_index=self.ignore_index,
                reduction='none',
                weight=self.class_weights.cuda(),
            )

            loss = loss.view(b, h, w)
            loss = loss.view(b, -1)

            if self.use_top_k:
                # Penalises the top-k hardest pixels
                k = int(self.top_k_ratio * loss.shape[1])
                loss, _ = torch.sort(loss, dim=1, descending=True)
                loss = loss[:, :k]

        return torch.mean(loss)

#---------------------------------------------------#
#  目标检测损失
#---------------------------------------------------#
class DetectionLoss(nn.Module):
    def __init__(self, anchors, nC, img_dim, anchor_idxs):
        super(DetectionLoss, self).__init__()
        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (1)
        self.bbox_attrs = 9 + nC
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nG = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).float()
        self.grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # self.weights = class_weights()

    def forward(self, p, targets=None, requestPrecision=False):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

        bs = p.shape[0]  # batch size
        nG = p.shape[2]  # number of grid points
        stride = self.img_dim / nG

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
        # self.weights = self.weights.cuda()

        # p.view(1, 30, 13, 13) -- > (1, 3, 13, 13, 10)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        P1_x = p[..., 0]  # Point1 x
        P1_y = p[..., 1]  # Point1 y
        P2_x = p[..., 2]  # Point2 x
        P2_y = p[..., 3]  # Point2 y
        P3_x = p[..., 4]  # Point3 x
        P3_y = p[..., 5]  # Point3 y
        P4_x = p[..., 6]  # Point4 x
        P4_y = p[..., 7]  # Point4 y

        pred_boxes = FT(bs, self.nA, nG, nG, 8)
        pred_conf = p[..., 8]  # Conf
        pred_cls = p[..., 9:]  # Class

        # Training
        if targets is not None:
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()
            SmoothL1Loss = nn.SmoothL1Loss()

            if requestPrecision:
                gx = self.grid_x[:, :, :nG, :nG]
                gy = self.grid_y[:, :, :nG, :nG]
                pred_boxes[..., 0] = P1_x.data + gx
                pred_boxes[..., 1] = P1_y.data + gy
                pred_boxes[..., 2] = P2_x.data + gx
                pred_boxes[..., 3] = P2_y.data + gy
                pred_boxes[..., 4] = P3_x.data + gx
                pred_boxes[..., 5] = P3_y.data + gy
                pred_boxes[..., 6] = P4_x.data + gx
                pred_boxes[..., 7] = P4_y.data + gy

            t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls, TP, FP, FN, TC = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nG,
                              requestPrecision)

            tcls = tcls[mask]
            if P1_x.is_cuda:
                t1_x, t1_y, t2_x, t2_y, t3_x, t3_y, t4_x, t4_y, mask, tcls = \
                    t1_x.cuda(), t1_y.cuda(), t2_x.cuda(), t2_y.cuda(), t3_x.cuda(), t3_y.cuda(), t4_x.cuda(), t4_y.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # Number of targets
            nM = mask.sum().float()  # Number of anchors (assigned to targets)
            nB = len(targets)  # Batch size
            k = nM / nB
            if nM > 0:
                lx1 = (k) * SmoothL1Loss(P1_x[mask], t1_x[mask]) / 8
                ly1 = (k) * SmoothL1Loss(P1_y[mask], t1_y[mask]) / 8
                lx2 = (k) * SmoothL1Loss(P2_x[mask], t2_x[mask]) / 8
                ly2 = (k) * SmoothL1Loss(P2_y[mask], t2_y[mask]) / 8
                lx3 = (k) * SmoothL1Loss(P3_x[mask], t3_x[mask]) / 8
                ly3 = (k) * SmoothL1Loss(P3_y[mask], t3_y[mask]) / 8
                lx4 = (k) * SmoothL1Loss(P4_x[mask], t4_x[mask]) / 8
                ly4 = (k) * SmoothL1Loss(P4_y[mask], t4_y[mask]) / 8

                lconf = (k * 10) * BCEWithLogitsLoss(pred_conf, mask.float())
                lcls = (k / self.nC) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
            else:
                lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, lcls, lconf = \
                    FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT(
                        [0]).requires_grad_(True), FT([0]).requires_grad_(True), \
                    FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT([0]).requires_grad_(True), FT(
                        [0]).requires_grad_(True), FT([0]).requires_grad_(True),

            # Sum loss components
            loss = lx1 + ly1 + lx2 + ly2 + lx3 + ly3 + lx4 + ly4 + lconf + lcls

            # Sum False Positives from unassigned anchors
            i = torch.sigmoid(pred_conf[~mask]) > 0.5
            if i.sum() > 0:
                FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()
            else:
                FPe = torch.zeros(self.nC)
            return loss, loss.item(), lconf.item(), lcls.item(), nT, TP, FP, FPe, FN, TC

        else:
            pred_boxes[..., 0] = P1_x + self.grid_x
            pred_boxes[..., 1] = P1_y + self.grid_y
            pred_boxes[..., 2] = P2_x + self.grid_x
            pred_boxes[..., 3] = P2_y + self.grid_y
            pred_boxes[..., 4] = P3_x + self.grid_x
            pred_boxes[..., 5] = P3_y + self.grid_y
            pred_boxes[..., 6] = P4_x + self.grid_x
            pred_boxes[..., 7] = P4_y + self.grid_y

            output = torch.cat((pred_boxes.view(bs, -1, 8) * stride,
                                torch.sigmoid(pred_conf.view(bs, -1, 1)),
                                pred_cls.view(bs, -1, self.nC)), -1)

            return output

#---------------------------------------------------#
#  depth pose autoencoder loss
#---------------------------------------------------#
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# 重构损失 loss_rec_autoencoder
def robust_l1(rec_img, org_img):
    eps = 1e-3
    return torch.sqrt(torch.pow(rec_img - org_img, 2) + eps ** 2)

# 重构 + 结构相似损失 loss_rec_ssim
def compute_reprojection_loss(pred, target,ssim=SSIM()):
    photometric_loss = robust_l1(pred, target).mean(1, True)
    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
    return reprojection_loss

# 张量梯度计算
def gradient(D):
    dy = D[:, :, 1:] - D[:, :, :-1]
    dx = D[:, :, :, 1:] - D[:, :, :, :-1]
    return dx, dy

# 平滑损失
# smooth1 loss_dis
# smooth2 loss_cvt
def get_smooth_loss(disp, img):
    b, _, h, w = disp.size()
    img = F.interpolate(img, (h, w), mode='area')

    disp_dx, disp_dy = gradient(disp)
    img_dx, img_dy = gradient(img)

    disp_dxx, disp_dxy = gradient(disp_dx)
    disp_dyx, disp_dyy = gradient(disp_dy)

    img_dxx, img_dxy = gradient(img_dx)
    img_dyx, img_dyy = gradient(img_dy)

    smooth1 = torch.mean(disp_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
              torch.mean(disp_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

    smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
              torch.mean(disp_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
              torch.mean(disp_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
              torch.mean(disp_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

    return smooth1, smooth2


# depth 平滑损失 loss_s
def get_smooth_loss_depth(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()