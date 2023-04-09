# YOLOv5 YOLO-specific modules

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())
logger = logging.getLogger(__name__)

from vega.networks.pytorch.detectors.yolov5.common import *
from vega.networks.pytorch.detectors.yolov5.experimental import *
from vega.networks.pytorch.detectors.yolov5.utils.general import make_divisible, check_file, set_logging
from vega.networks.pytorch.detectors.yolov5.utils.torch_utils import initialize_weights, select_device

class YOLOBACKBONE(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super(YOLOBACKBONE, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels #! 对应key不存在的时候，返回默认值，也就是3
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value 重新赋值yaml对应的nc的值
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value 重新赋值anchor
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist #^ 准确的说，是模型序列列表和保存的相关分支的索引号
        self.save = [2, 4, 6 ,10]
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names 默认数学转换的名字
        self.inplace = self.yaml.get('inplace', True) #! 对应key不存在的时候，返回默认值
        initialize_weights(self)

    def forward(self, x, output_feat = False):
        y, dt = [], []
        for m in self.model:
            x = m(x)
            if m.i in self.save:
                y.append(x)
        if output_feat:
            return y
        else:
            return x

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors 每个层有多少个anchor
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) 类别，confidence，x，y，w，h

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out 这里只是将相关的部件解析出来，并没有进行网络拓扑接口的连接
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings 执行string得到特定写的层如Focus,Conv,C3等
        for j, a in enumerate(args): #! 除了SPP这个特定的层，args里面的参数基本上就只有一个
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings 执行string得到实际list
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain 深度的增益 #! 这里的增益只会针对大于1的相关参数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0] #! ch[f]只有在特定层的时候，f才是非-1的其他值，而args[0]的值代表的是输出的整个通道的数量
            if c2 != no:  # if not output 最后的输出层
            # if i !=9:  # if not output 最后的输出层
                c2 = make_divisible(c2 * gw, 8) #! 保证能被8整除

            args = [c1, c2, *args[1:]] # c1输入，c2输出，一般其他参数就是(filter，stride)
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats 这里的重复，指代的是残差块重复的次数，可参考yolov5的结构图
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f]) # 每一个维度concat起来之后的维度
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        if i ==10:
            args[1] = 512
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module #! 类似于nn.Upsample的模块就直接进行运算了，没有经过上面的if语句
        t = str(m)[8:-2].replace('__main__.', '')  # module type #! 这里很值得注意看
        np = sum([x.numel() for x in m_.parameters()])  # number params #! 获取整个当前层序列所对应的参数
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist 等于-1的部分是不添加的 保存的都是索引吧！(for head part!)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2) #! 上一次输出通道的值
    return nn.Sequential(*layers), sorted(save) #! 针对yolov5s的话，相关的sava排序之后，就是[4,6,10,14,17,20,23]都是分支拉出来的部分

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default='/home/zhandongxu/Code/vega_mt/models/yolov5s.yaml',
                        help='model.yaml')

    parser.add_argument('--device',
                        default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = YOLOBACKBONE(opt.cfg).to(device)
    model.eval()

    # Profile
    img = torch.rand(8, 3, 288, 512).to(device)

    import time
    for i in range(1000):
        tic = time.time()
        torch.cuda.synchronize()
        y = model(img, output_feat=True)
        torch.cuda.synchronize()
        print("processing time is %.2f" %(1000*(time.time() - tic)))
    for index in range(len(y)):
        print(y[index].shape)
