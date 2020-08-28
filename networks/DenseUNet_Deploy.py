import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('utils')
sys.path.append('networks/backbone')
from collections import OrderedDict
from networks.utils import get_norm
from Layers import *
from Layers import (_OutConv)
from networks.backbone.densenet import (_DenseBlock, _DenseLayer, _Transition, _UpBlock)
from Global_pool import GlobalPool


class DenseUNet_Deploy(nn.Module):
    def __init__(self, n_channels, n_classes, t_classes, pool, bilinear=True):
        super(DenseUNet_Deploy, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.t_classes = t_classes
        self.bilinear = bilinear
        self.pool = pool
        self.global_pool = GlobalPool(pool)


        # DenseNet Arch Hyperparameters
        growth_rate = 32
        block_config = [6, 12, 24, 16]
        num_init_features = 64
        norm_type = 'BatchNorm'
        bn_size = 4
        drop_rate = 0


        # Source Domain Architecture
        # First convolution
        self.s_inc = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.n_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # noqa
            ('norm0', get_norm(norm_type, num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # S - Denseblock 1 - 128 x 128
        num_features = num_init_features # 64
        self.s_denseblock_1 = _DenseBlock(num_layers=block_config[0],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[0] * growth_rate # 256
        self.s_transition_1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                            norm_type=norm_type)
        num_features = num_features // 2 # 128

        # S - Denseblock 2 - 64 x 64
        self.s_denseblock_2 = _DenseBlock(num_layers=block_config[1],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[1] * growth_rate # 512
        self.s_transition_2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                            norm_type=norm_type)
        num_features = num_features // 2 # 256

        # S - Denseblock 3 - 32 x 32
        self.s_denseblock_3 = _DenseBlock(num_layers=block_config[2],
                                          num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                                          growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[2] * growth_rate # 1024
        self.s_transition_3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                          norm_type=norm_type)
        num_features = num_features // 2 # 512

        # S - Denseblock 4 - 16 x 16
        self.s_denseblock_4 = _DenseBlock(num_layers=block_config[3],
                                          num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                                          growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + block_config[3] * growth_rate # 1024
        self.s_transition_4 = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                          norm_type=norm_type)
        num_features = num_features // 2  # 512

        # S - UpBlock 4 - 512 x 8 x 8
        in_channels = num_features * 2 # 512 + 512 = 1024
        out_channels = num_features # 512
        self.s_upblock_4 = _UpBlock(in_channels, out_channels, norm_type=norm_type, bilinear=self.bilinear)

        # S - UpBlock 3 - 512 x 16 x 16
        in_channels = out_channels // 2 + out_channels # 256 + 512 = 768
        out_channels = out_channels // 2 # 256
        self.s_upblock_3 = _UpBlock(in_channels, out_channels, norm_type=norm_type, bilinear=self.bilinear)

        # S - UpBlock 2 - 256 x 32 x 32
        in_channels = out_channels // 2 + out_channels # 128 + 256 = 384
        out_channels = out_channels // 2 # 128
        self.s_upblock_2 = _UpBlock(in_channels, out_channels, norm_type=norm_type, bilinear=self.bilinear)

        # S - UpBlock 1 - 128 x 64 x 64
        in_channels = out_channels // 2 + out_channels # 64 + 128 = 192
        out_channels = out_channels // 2 # 64
        self.s_upblock_1 = _UpBlock(in_channels, out_channels, norm_type=norm_type, bilinear=self.bilinear)

        # S - OutConv - 64 x 128 x 128
        in_channels = out_channels
        out_channels = self.n_classes
        self.s_outc = _OutConv(in_channels, out_channels) # num_classes x 512 x 512


        # Target Domain Architecture
        # T - First convolution
        self.t_inc = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # noqa
            ('norm0', get_norm(norm_type, num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # T - Denseblock 1 - 128 x 128
        num_features = num_init_features * 2 # 64 + 64 = 128
        self.t_denseblock_1 = nn.Sequential(OrderedDict([('denseblock1', _DenseBlock(num_layers=block_config[0],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate))]))
        num_features = num_features + block_config[0] * growth_rate # 128 + 192 = 320
        self.t_transition_1 = nn.Sequential(OrderedDict([('transition1', _Transition(num_input_features=num_features,
                            num_output_features=num_init_features * 2, norm_type=norm_type))]))
        num_features = num_init_features * 4 # 128 + 128 = 256

        # T - Denseblock 2 - 64 x 64
        self.t_denseblock_2 = nn.Sequential(OrderedDict([('denseblock2', _DenseBlock(num_layers=block_config[1],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate))]))
        num_features = num_features + block_config[1] * growth_rate # 256 + 384 = 640
        self.t_transition_2 = nn.Sequential(OrderedDict([('transition2', _Transition(num_input_features=num_features,
                            num_output_features=num_init_features * 4, norm_type=norm_type))]))
        num_features = num_init_features * 8 # 256 + 256 = 512

        # T - Denseblock 3 - 32 x 32
        self.t_denseblock_3 = nn.Sequential(OrderedDict([('denseblock3', _DenseBlock(num_layers=block_config[2],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate))]))
        num_features = num_features + block_config[2] * growth_rate # 512 + 768 = 1280
        self.t_transition_3 = nn.Sequential(OrderedDict([('transition3', _Transition(num_input_features=num_features,
                            num_output_features=num_init_features * 8, norm_type=norm_type))]))
        num_features = num_init_features * 16 # 512 + 512 = 1024

        # T - Denseblock 4 - 16 x 16
        self.t_denseblock_4 = nn.Sequential(OrderedDict([('denseblock4', _DenseBlock(num_layers=block_config[3],
                            num_input_features=num_features, norm_type=norm_type, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate))]))
        num_features = num_features + block_config[3] * growth_rate # 1024 + 512 = 1536
        self.t_transition_4 = _Transition(num_input_features=num_features, num_output_features=num_features // 3,
                                          norm_type=norm_type)
        num_features = num_features // 3  # 512

        # T - Final batch norm - (512 + 512) x 8 x 8
        self.t_norm_5 = nn.Sequential(OrderedDict([('norm5', get_norm(norm_type, num_features * 2))]))

        # T - Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.t_classifier = nn.Conv2d(num_features * 2, self.t_classes, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x_t):

        # target domain
        self.x1_ts = self.s_inc(x_t) # 64 x 128 x 128
        self.x2_ts = self.s_denseblock_1(self.x1_ts)
        self.x2_ts = self.s_transition_1(self.x2_ts) # 128 x 64 x 64
        self.x3_ts = self.s_denseblock_2(self.x2_ts)
        self.x3_ts = self.s_transition_2(self.x3_ts) # 256 x 32 x 32
        self.x4_ts = self.s_denseblock_3(self.x3_ts)
        self.x4_ts = self.s_transition_3(self.x4_ts) # 512 x 16 x 16
        self.x5_ts = self.s_denseblock_4(self.x4_ts)
        self.x5_ts = self.s_transition_4(self.x5_ts)  # 512 x 8 x 8

        self.x1_t = self.t_inc(x_t) # 64 x 128 x 128
        self.x1_t = torch.cat([self.x1_t, self.x1_ts], dim=1) # 128 x 128 x 128

        self.x2_t = self.t_denseblock_1(self.x1_t) # 320 x 128 x 128
        self.x2_t = self.t_transition_1(self.x2_t) # 128 x 64 x 64
        self.x2_t = torch.cat([self.x2_t, self.x2_ts], dim=1)  # 256 x 64 x 64

        self.x3_t = self.t_denseblock_2(self.x2_t) # 640 x 64 x64
        self.x3_t = self.t_transition_2(self.x3_t) # 256 x 32 x 32
        self.x3_t = torch.cat([self.x3_t, self.x3_ts], dim=1)  # 512 x 32 x 32

        self.x4_t = self.t_denseblock_3(self.x3_t) # 1280 x 32 x 32
        self.x4_t = self.t_transition_3(self.x4_t) # 512 x 16 x 16
        self.x4_t = torch.cat([self.x4_t, self.x4_ts], dim=1)  # 1024 x 16 x 16

        self.x5_t = self.t_denseblock_4(self.x4_t) # 1536 x 16 x 16
        self.x5_t = self.t_transition_4(self.x5_t)  # 512 x 8 x 8
        self.x5_t_f = torch.cat([self.x5_t, self.x5_ts], dim=1)  # 1024 x 8 x 8

        self.feature = self.t_norm_5(self.x5_t_f) # 1024 x 8 x 8
        self.feature = self.avgpool(self.feature) # 1024 x 1 x 1

        logits_t = self.t_classifier(self.feature) # t_classes x 1 x 1
        logits_t = logits_t.squeeze(-1).squeeze(-1) # t_classes

        return logits_t

