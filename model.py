import os
import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F

import argparse

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, \
                bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, \
                stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, \
                kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, \
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride= 1, padding=1)
        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride= 1, padding=1)
        # add P4 elementwise to C3
        self.P3_1           = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride= 1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6             = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1           = nn.ReLU()
        self.P7_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5      = self.P5_1(C5)
        P5_up   = self.P5_upsampled(P5)
        P5      = self.P5_2(P5)

        P4      = self.P4_1(C4)
        P4      = P4 + P5_up
        P4_up   = self.P4_upsampled(P4)
        P4      = self.P4_2(P4)

        P3      = self.P3_1(C3)
        P3      = P3 + P4_up
        P3      = self.P3_2(P3)

        P6      = self.P6(C5)
        P7      = self.P7_1(P6)
        P7      = self.P7_2(P7)

        return [P3, P4, P5, P6, P7]

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,padding=1)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,padding=1)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,padding=1)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,padding=1)
        self.act4 = nn.ReLU(inplace=True)

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out = [B x C x W x H] (C = 4*num_anchors)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out = [B x C x W x H] (C = n_classes + n_anchors)
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class ResNet(nn.Module):
    def __init__(self,num_classes, net_type='FPN50', block=Bottleneck):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if net_type == 'FPN50':
            num_blocks = [3,4,6,3]
        elif net_type == 'FPN101':
            num_blocks = [3,4,23,3]

        # input channel = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,   64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block,  128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,  256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block,  512, num_blocks[3], stride=2)
        fpn_sizes = [self.layer2[num_blocks[1]-1].conv3.out_channels, self.layer3[num_blocks[2]-1].conv3.out_channels, self.layer4[num_blocks[3]-1].conv3.out_channels]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.regressionModel= RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes = num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, inputs):
        # Bottom-up
        if self.training:
            img_batch = inputs
        else:
            img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        loc_preds = [self.regressionModel(feature) for feature in features]
        cls_preds = [self.classificationModel(feature) for feature in features]

        return loc_preds, cls_preds

def FPN50(num_classes):
    return ResNet(num_classes, 'FPN50')

def FPN101(num_classes):
    return ResNet(num_classes, 'FPN101')

# INITIALIZE
net_func = {
        'FPN50': FPN50,
        'FPN101': FPN101
        }

def build_net(net_type, loss_fn, data_name):
    # Build RetinaNet module
    save_path = './init_weight/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print('Loading resnet into FPN..')

    if data_name == 'VOC' and loss_fn == 'sigmoid':
        num_classes = 20
    elif data_name == 'VOC' and loss_fn == 'softmax':
        num_classes = 21
    elif data_name == 'COCO' and loss_fn == 'sigmoid':
        num_classes = 80
    if data_name == 'COCO' and loss_fn == 'softmax':
        num_classes = 81
    elif data_name == 'KITTI' and loss_fn == 'sigmoid':
        num_classes = 3
    if data_name == 'KITTI' and loss_fn == 'softmax':
        num_classes = 4

    # load resnet weight

    if net_type is 'PIXOR':
        pass
    else: 
        print('Construct RetinaNet..')
        net = net_func[net_type](num_classes)
        for mod in net.modules():
            # Initialization p5
            if isinstance(mod, nn.Conv2d):
                init.normal_(mod.weight, mean=0, std=0.01)
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

            # Because of torch initialize BN w~U[0,1], b=0 value initialize w=1 again
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()

        # pre-trained on ImageNet
        if net_type == 'FPN50':
            res = models.resnet50(True)
        elif net_type == 'FPN101':
            res = models.resnet101(True)
        res_wgt = res.state_dict()
        net_wgt = net.state_dict()
        for k in res_wgt.keys():
            if not k.startswith('fc'):
                net_wgt[k] = res_wgt[k]

        # Initizliation p5
        net.load_state_dict(net_wgt)
        pi = 0.01
        init.constant_(net.classificationModel.output.bias, -math.log((1-pi)/pi))
        if loss_fn == 'softmax':
            net.classificationModel.output.bias.data[[0,21,42,63,84,105,126,147,168]] = np.log(20*(1-pi)/pi)
        torch.save(net.state_dict(), save_path+'net_'+args.data+'_'+args.loss_fn+'_'+net_type+'.pt')
        print('Success')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RetinaNet network build parser')
    parser.add_argument('--network', '-n', default='FPN50', choices=['FPN50', 'FPN101', 'PIXOR'], \
            type=str, help='FPN50 || FPN101 || PIXOR')
    parser.add_argument('--loss_fn', '-loss', default='sigmoid', choices=['sigmoid', 'softmax'], \
            type=str)
    parser.add_argument('--data', '-data', default='COCO', choices=['VOC', 'COCO', 'KITTI'], type=str)
    args = parser.parse_args()

    build_net(args.network, args.loss_fn, args.data)
