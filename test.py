from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import argparse
import sys
sys.path.append('utils/')
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import auc

from networks import DenseUNet_Deploy
from datasets.get_dataset import get_ODIR
from utils.Layers import *
from utils.Losses import *
from utils.seg_eval_metrics import *
from utils.utils import *
from utils import Logger, AverageMeter, odir_metrics, mkdir_p, savefig


parser = argparse.ArgumentParser(description='Inductive Transfer Learning - Multi-scale Transfer Connections & Domain-Specific Adversarial Learning.')

parser.add_argument('--image_path_source', default='/extendspace/yizhou/DR/FGADR-2842/Segmentation/demonstration', type=str, help='root path to source domain image folder')
parser.add_argument('--image_path_target', default='/extendspace/yizhou/DR/ODIR-5K', type=str, help='root path to target domain image folder')
parser.add_argument('--input_size', default=512, type=int, help='the input size of images')
parser.add_argument('--network', default='dense_unet', type=str, help='network architecture: 2D_unet, dense_unet')
parser.add_argument('--global_pool', default='PCAM', type=str, help='global pooling method for CAM')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--base_lr', default=0.001, type=float, help='base learning rate for SGD optimizer')
parser.add_argument('--optimizer', default='Adam', type=str, help='SGD / Adam')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD')
parser.add_argument('--ckpt_dir', default='./save_models/ALL', type=str, help='ckpt path')
parser.add_argument('--pretrain', default='epoch-283-kappa-0.7389.pth.tar', type=str, help='path to the pretrained model')
parser.add_argument('--num_epochs', default=300, type=int, help='training epochs')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='Support one GPU & multiple GPUs')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes of source domain')
parser.add_argument('--t_classes', default=8, type=int, help='number of classes of target domain')
parser.add_argument('--class_idx', default=4, type=int, help='the index of class for training [0: ma, 1: ex, 2: se, 3: he, 4: all]')
parser.add_argument('--augmentation', default=True, type=bool, help='whether to do data augmentation')
parser.add_argument('--split_index', default=5, type=int, help='train_test_split index')
parser.add_argument('--save_path', default='./save_images', type=str, help='root path to save image folder')
args = parser.parse_args()

args.num_classes = [1, 1, 1, 1]


def val(data_loader, model, class_weights_t):
    end = time.time()

    n_val = len(data_loader)
    losses_cls = AverageMeter()
    score_kappa = AverageMeter()
    score_auc = AverageMeter()
    score_f1 = AverageMeter()
    acc_cls = np.zeros(args.t_classes)

    with torch.no_grad():
        for i, (image, targets) in enumerate(data_loader):
            print(i, '/', n_val)

            if args.gpu is not None:
                image, targets = image.cuda(), targets.cuda()

            output_t = model(image)

            criterion = nn.BCEWithLogitsLoss(weight=class_weights_t.cuda())
            loss_cls = criterion(output_t, targets)

            # Evaluation
            predicts = nn.Sigmoid()(output_t)
            kappa, auc, f1, final, acc_each_label = odir_metrics(predicts.data, targets.data)
            acc_cls += acc_each_label
            losses_cls.update(loss_cls.data, image.size(0))
            score_kappa.update(kappa)
            score_auc.update(auc)
            score_f1.update(f1)

    val_time = time.time() - end
    print('Validation Target Domain: Cls Average Loss: %2.5f, Kappa: %3.5f, AUC: %3.5f, F1: %3.5f' % (losses_cls.avg,
                        score_kappa.avg, score_auc.avg, score_f1.avg))

    acc_cls = acc_cls / n_val
    print(acc_cls)

    return score_kappa.avg, score_auc.avg, score_f1.avg


def main(args):
    if args.gpu is not None:
        print(('Using GPU: ' + args.gpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')

    random.seed(1130)
    torch.manual_seed(1130)
    torch.cuda.manual_seed_all(1130)
    cudnn.benchmark = True

    print(args)

    print('Creating model...')
    if args.network == '2D_unet':
        model = UNet2D(n_channels=3, n_classes=args.n_classes, num_classes=args.num_classes, pool=args.global_pool, bilinear=True)
    elif args.network =='dense_unet':
        model = DenseUNet_Deploy(n_channels=3, n_classes=args.n_classes,
                          t_classes=args.t_classes, pool=args.global_pool, bilinear=True)
    else:
        print('TODO')


    print('Loading Weights...')
    all_model_dir = os.path.join(args.ckpt_dir, args.pretrain)
    if args.class_idx == 4:
        model = load_model(model, all_model_dir)
    else:
        print('TODO')

    if args.gpu is not None:
        model = nn.DataParallel(model).cuda()

    print('Setting up target domain data...')
    trainset_t, valset_t, class_weights_t = get_ODIR(dataset_path=args.image_path_target, input_size=args.input_size,
                                                     batch_size=args.batch_size, split_index=args.split_index)

    val_loader_t = data.DataLoader(valset_t, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print('Start Testing...')

    kappa, auc, f1 = val(val_loader_t, model, class_weights_t)

    print('Testing Finished.')


if __name__ == '__main__':
    main(args)
