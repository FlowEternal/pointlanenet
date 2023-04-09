# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss() 不进行规约的操作
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4)) #! 这个因子会等比扩大pred和true差距大的类别的权重，而缩小差距小的类别的权重(真的有意思)
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module): #! 以Module的形式去构建了相当于一个层
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss() #! 因为FocalLoss本质上是只支持BCE的，对于CE，FocalLoss其计算的方式是与论文不保持一致的
        self.gamma = gamma
        self.alpha = alpha #! 论文取值分别是alpha=0.25和gamma=2
        self.reduction = loss_fcn.reduction #^ 保留原始的归约方式
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element #! 这里取none是因为，需要每一个二分类，如果没有，取mean或者sum是有问题的

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py #* 哈哈哈，是tensorflow_addons，哈哈哈~阔以
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob) # 针对每个类别的二分类交叉熵
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha) # 各个二分类参数
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor # yeap! 就是这样

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        #! 这里之前一直没有考虑过BCE和CE的区别，现在想起来，BCE前面是sigmod，而CE前面是softmax(一般，大概~)，计算的方式方法完全不一样，但是最后的结果却差不太多
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device)) 
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets 如果有label_smoothing这个参数就取，不然就直接是0

        # Focal loss #! 之前一直没有理解focal loss的BCE,现在应该仔细去理解相关原因(基于BCE实际上就是对每个单类做Focal Loss) 真的有意思
        g = h['fl_gamma']  # focal loss gamma 
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7 如果det.nl是3，就返回[4.0, 1.0, 0.4]，不然就是[4.0, 1.0, 0.25, 0.06, .02]
        #? 上面的self.balance的作用到底是啥~
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index 求索引，对应yolov5s应该是1，因为stride是[8,16,32]
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance #^ model.gr是模型iou loss ratio
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets #? 这里没明白啊
        #* A. 针对yolov5s tcls分别是8,16,32倍对应框的类别
        #*                tbox分别是8,16,32倍对应的相关框的回归参数 分别对应 b_x,b_y,b_w,b_h
        #*                indices分别是8,16,32倍的 batch idx，anchor indices，gird index y和gird index x
        #*                anchors分别是8,16,32倍的对应锚框大小

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets #! 这里获取对应标签位置在该特征图上的特征向量，格式为n x 85 (yolov5s)

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5 #!!!! 这里非常重要，这也是yolov5不同的地方，由于其采用了跨网格预测，故xy预测输出不再是0-1，而是-1～1，加上offset偏移，则为-0.5-1.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i] #! 宽高与上面同理
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target) 计算的IOU的大小(IOU,GIOU,DIOU,CIOU)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio #^ self.gr是模型iou loss ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets #^ self.cn是smooth后的最小参数
                    t[range(n), tcls[i]] = self.cp #^ 对应的类别设置成为1或者smooth后的类别
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE 当前层级下的BCE loss

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss #! 这里的balance强调了不同层级的损失比例，对于低倍下采样其采样权重高，也就是对偏小的目标有更强的倚重，这对于yolo系列来说还是有点意思的
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item() #! 动态修正autobalance的相关参数

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    #! 这里基本是整个yolov5比较核心也比较难以理解的部分，一定要好好参考 https://zhuanlan.zhihu.com/p/183838757
    def build_targets(self, p, targets): #^ 这里的p是具有[8,16,32]倍下采样的feature map,原始targets的size是 nt x 6 (nt代表标签个数)
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets nt代表标签框的个数 targets的格式为 [index,confidence,x,y,w,h] xywh的范围是[0,1]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) 格式为 na x nt (行分别是0到na的整数) 
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices 
        #^ 上面targets.repeat后成为了 na x nt x 6 的格式, ai[:, :, None]后格式为 na x nt x 1，当用第2个维度进行concat的时候，其成为 na x nt x 7
        #^ na x nt x 7分别代表的是，单层映射的锚框个数、当前图像存在的标签个数、7维的特征向量(index,class,x,y,w,h,ai分配的这个单层anchor对应索引) 

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets 偏移量为0.5个特征图

        for i in range(self.nl): #! 针对每一层锚框
            anchors = self.anchors[i] #^ 格式为 na x 2
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain #^ torch.tensor(p[i].shape)输出的格式为 BCHW , 因此gain[2:6]的tensor实际上是W H W H

            # Match targets to anchors
            t = targets * gain #! 这样原始的x,y,w,h就从比例变为了对应层feature map的位置
            if nt: #^ 保证有框选的类别
                # Matches (滤除当前不太匹配的锚框)
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio #! 这里t的size是na x nt x 2，anchors的size是 na x 1 x 2,最后依旧得到na x nt x 2
                #! 上面的部分相当于用各个对应的框去分别除以对应的anchor锚框size，得到的是各个锚框下的比例
                #! 还需要注意的是，上面的anchors已经是各个下采样特征图下的比例，参考Model可以看到
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare #^ 锚框与实际框的差距比例,[0]是只取最大的值value而不是indice
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter #^ 只找到那些差距比例比较小的框(避免大的差距比例，从而避免框的回归拟合过程因差距过大而发生振荡,本质上也就是分配最相近的框进行检测),也就是当前层级检测的目标
                #& 话说，这个和使用iou进行过滤貌似是一致的啊，只是这里采用的过滤方式是用的比例(也是因为由于是比例，一个box可能与多个anchor匹配，因此变相的增加了正样本的数量，这比max iou有了很大的不一样)
                #! 特别注意，上面的t已经变成了满足过滤条件的 nt_filter x 7 格式了
                #! 其实上面就是针对与一个label，在3个anchor中，删除偏差过大的那个anchor匹配，保留的都是在anchor_t比例差距下面数据

                # Offsets  #? (滤除边缘部分的框?) A.增加两个框的正样本
                gxy = t[:, 2:4]  # grid xy 标签中心点的位置 nt_filter x 2
                gxi = gain[[2, 3]] - gxy  # inverse 实际宽高减去中心点的反向位置 nt_filter x 2
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T # nt_filter
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T # nt_filter
                #! 上边是找到在整个特征图内部的部分区域偏移(具体位置请自行理解)
                j = torch.stack((torch.ones_like(j), j, k, l, m)) #^ 格式为 5 x nt_filter 这里torch.ones_like(j)返回的依旧类别是BOOL
                t = t.repeat((5, 1, 1))[j] # 这里重复之前的t是经过了过滤后的，各个anchor拟合情况较好的框(拟合对应的框由最后一个维度的有一个index表示)
                #? 为什么最外层要重复5次呢，而且仅仅排除了 t.repeat((5, 1, 1))的格式为 5 x nt_filter x 7 ？
                #* A.因为得到的j是一个5 x nx的Bool Tensor，其中第一行代表全为True(表示保留原始数据)，其他可以绘制feature map去找到对应的空间区域，然后分别滤除选择满足情况的区域
                #! A.从本质上来讲，j与l和k与m是互斥的，上面的操作，j的第一行的全True表示的是保留原始框，而其他4行的目的，是为了获取最近的两个grid位置，然后同样的框做为正样本，这样极大的
                #! 引入了正样本，但是，这样额外的两个框与原来gt的iou如果程度不太好，这样会导致认定为bad case 详细参考 https://zhuanlan.zhihu.com/p/183838757
                #? 有没有可能，前面的几十个epoch进行粗犷的这种方式训练，而等到一定精度后的epoch，还是采用yolov3的max iou匹配模式进行精细化控制？
                #^ 上面t.repeat((5, 1, 1))的格式为 5 x nt_filter x 7
                #^ j的格式为 5 x nt_filter
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] #! 这里的offsets表示的是t对应的偏移的数
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class 当前batch的idx与对应的类别
            gxy = t[:, 2:4]  # grid xy 得到正样本的框的中心点
            gwh = t[:, 4:6]  # grid wh 得到正样本的框的长宽
            gij = (gxy - offsets).long() # 对应的检测的索引
            gi, gj = gij.T  # grid xy indices x坐标与y坐标

            # Append
            a = t[:, 6].long()  # anchor indices  #! 对应的anchor索引
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices 其中clamp_保证其在这个feature map空间范围内
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 分别对应 b_x,b_y,b_w,b_h
            anch.append(anchors[a])  # anchors 对应的anchor的大小
            tcls.append(c)  # class 对应的类别

        return tcls, tbox, indices, anch
