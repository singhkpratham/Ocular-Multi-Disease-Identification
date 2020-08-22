import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('utils')
from Layers import *
from Global_pool import GlobalPool


class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes, num_classes, pool, bilinear=True):
        super(UNet2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.pool = pool
        self.global_pool = GlobalPool(pool)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor)
        # self.classifier = nn.Linear(1024, n_classes)
        self._init_classifier()
        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)
        self.outc = OutConv(64, 1)


    def _init_classifier(self):
        for index, num_class in enumerate(self.num_classes):
            setattr(self, "fc_" + str(index),
                    nn.Conv2d(
                        512,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()


    def forward_one(self, x):
        x = self.up1(x, self.x4)
        x = self.up2(x, self.x3)
        x = self.up3(x, self.x2)
        x = self.up4(x, self.x1)
        x = self.outc(x)
        return x


    def forward(self, x):
        self.x1 = self.inc(x)
        self.x2 = self.down1(self.x1)
        self.x3 = self.down2(self.x2)
        self.x4 = self.down3(self.x3)
        self.x5 = self.down4(self.x4)
        # outcls = F.adaptive_avg_pool2d(x5, (1, 1))
        # outcls = torch.flatten(outcls, 1)
        # logits_cls = self.classifier(outcls)

        # [(N, 1), (N,1),...]
        logits_cls = list()
        # [(N, 1, H, W), (N, 1, H, W),...]
        logits_map = list()
        weights_map = list()
        # [(N, H_input, W_input), (N, H_input, W_input), ...]
        logits_mask = list()
        # [(N, C, H, W), (N, C, H, W), ...]
        weighted_feats = list()
        for index, num_class in enumerate(self.num_classes):

            classifier = getattr(self, "fc_" + str(index))

            # (N, 1, H, W)
            logit_map = None
            if not (self.pool == 'AVG_MAX' or
                    self.pool == 'AVG_MAX_LSE'):
                logit_map = classifier(self.x5)
                logits_map.append(logit_map)
            # feat - (N, C, 1, 1); weighted_feat - (N, C, H, W)
            if self.pool == 'PCAM':
                # print('Using PCAM')
                feat, weighted_feat, weight_map = self.global_pool(self.x5, logit_map)
                weights_map.append(weight_map)
            elif self.pool == 'AVG':
                # print('Using AVG')
                feat = self.global_pool(self.x5, logit_map)
                cls_kernels = classifier.weight
                cls_bias = classifier.bias
                output_manual = []
                for idx in range(512):
                    cls_kernel = cls_kernels[:, idx:idx+1] # (out_channel, in_channels/groups, kH, kW)
                    # print(cls_kernel.size())
                    input = self.x5[:, idx:idx+1] # (minibatch, in_channels, iH, iW)
                    # print(input.size())
                    out = F.conv2d(input, cls_kernel) + cls_bias
                    output_manual.append(out)
                weighted_feat = torch.cat(output_manual, dim=1)
                # print(weighted_feat.size())

            feat = F.dropout(feat, p=0.5, training=self.training)

            # (N, num_class, 1, 1)
            logit_cls = classifier(feat)
            # (N, num_class, 1, 1)
            logit_cls = logit_cls.squeeze(-1).squeeze(-1)
            logits_cls.append(logit_cls)

            weighted_feats.append(weighted_feat)

        # print(weighted_feats[0].max())
        # print(weighted_feats[1].max())
        # print(weighted_feats[2].max())
        # print(weighted_feats[3].max())

        logits_0 = self.forward_one(weighted_feats[0])
        logits_1 = self.forward_one(weighted_feats[1])
        logits_2 = self.forward_one(weighted_feats[2])
        logits_3 = self.forward_one(weighted_feats[3])

        print('logits_0', logits_0.max())
        print('logits_1', logits_1.max())
        print('logits_2', logits_2.max())
        print('logits_3', logits_3.max())

        logits_mask = [logits_0, logits_1, logits_2, logits_3]

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        return logits_mask, logits_cls, logits_map

