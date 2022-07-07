import torch.nn as nn
from torch.cuda.amp import autocast

from .backbones import *
from .gem_pooling import GeneralizedMeanPoolingP

__factory__ = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet152_ibn_a': resnet152_ibn_a,
}


class base_net(nn.Module):
    def __init__(self, backbone_name, last_stride=1, end_layer=2, pretrained=True):
        super(base_net, self).__init__()
        model = __factory__[backbone_name](last_stride, pretrained)
        self.end_layer = end_layer
        self.out_channels = {'1': 256, '2': 512, '3': 1024, '4': 2048}[str(end_layer)]
        layers = [nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)]
        if end_layer >= 1:
            layers.append(model.layer1)
        if end_layer >= 2:
            layers.append(model.layer2)
        if end_layer >= 3:
            layers.append(model.layer3)
        if end_layer >= 4:
            layers.append(model.layer4)
        self.layers = nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        x = self.layers(x)
        return x


class id_net(nn.Module):
    def __init__(self, backbone_name, num_classes, last_stride=1, start_layer=3, pooling_type='avg', pretrained=True):
        super(id_net, self).__init__()
        model = __factory__[backbone_name](last_stride, pretrained)
        self.start_layer = start_layer
        pooling_layer = \
        {'avg': nn.AdaptiveAvgPool2d(1), 'max': nn.AdaptiveMaxPool2d(1), 'gem': GeneralizedMeanPoolingP()}[pooling_type]
        if start_layer == 0:
            self.layers = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                        model.layer1, model.layer2, model.layer3, model.layer4,
                                        pooling_layer)
        elif start_layer == 1:
            self.layers = nn.Sequential(model.layer1, model.layer2, model.layer3, model.layer4, pooling_layer)
        elif start_layer == 2:
            self.layers = nn.Sequential(model.layer2, model.layer3, model.layer4, pooling_layer)
        elif start_layer == 3:
            self.layers = nn.Sequential(model.layer3, model.layer4, pooling_layer)
        elif start_layer == 4:
            self.layers = nn.Sequential(model.layer4, pooling_layer)
        else:
            raise AttributeError(
                'supported start layer should be in list [0,1,2,3,4], but got start layer %d' % start_layer)
        self.bn = nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    @autocast()
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        if not self.training:
            return x
        logits = self.classifier(x)
        return x, logits


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
