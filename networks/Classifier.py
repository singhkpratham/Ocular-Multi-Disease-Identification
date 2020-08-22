import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('utils')
from Layers import *
from Global_pool import GlobalPool

from networks.backbone.vgg import (vgg19, vgg19_bn)
from networks.backbone.densenet import (densenet121, densenet169, densenet201)
from networks.backbone.inception import (inception_v3)

BACKBONES = {'vgg19': vgg19,
             'vgg19_bn': vgg19_bn,
             'densenet121': densenet121,
             'densenet169': densenet169,
             'densenet201': densenet201,
             'inception_v3': inception_v3}


BACKBONES_TYPES = {'vgg19': 'vgg',
                   'vgg19_bn': 'vgg',
                   'densenet121': 'densenet',
                   'densenet169': 'densenet',
                   'densenet201': 'densenet',
                   'inception_v3': 'inception'}


class Classifier(nn.Module):

    def __init__(self, args):
        super(Classifier, self).__init__()
        self.n_classes = args.n_classes
        self.num_classes = args.num_classes
        self.backbone = args.backbone
        self.backbone_model = BACKBONES[args.backbone](args)
        self.pool = args.global_pool
        self.global_pool = GlobalPool(args.global_pool)
        self.expand = 1
        self.fc_bn = args.fc_bn
        self._init_classifier()
        self._init_bn()

    def _init_classifier(self):
        for index, num_class in enumerate(self.num_classes):
            if BACKBONES_TYPES[self.backbone] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.backbone] == 'densenet':
                setattr(
                    self,
                    "fc_" +
                    str(index),
                    nn.Conv2d(
                        self.backbone_model.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.backbone] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.backbone)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.num_classes):
            if BACKBONES_TYPES[self.backbone] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.backbone] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone_model.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.backbone] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone type : {}'.format(self.backbone)
                )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone_model(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.num_classes):

            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            if not (self.pool == 'AVG_MAX' or
                    self.pool == 'AVG_MAX_LSE'):
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat, _ = self.global_pool(feat_map, logit_map)

            if self.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
            feat = F.dropout(feat, p=0.5, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        return logits, logit_maps

