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

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.init as init
from sklearn.metrics import auc

from networks import UNet2D
from datasets.get_dataset import get_FGADR_Seg_test
from utils.Layers import *
from utils.Losses import *
from utils.seg_eval_metrics import *
from utils.utils import *


parser = argparse.ArgumentParser(description='Train Semantic Segmentation Model')

parser.add_argument('--image_path_source', default='/extendspace/yizhou/DR/FGADR-2842/Segmentation/demonstration', type=str, help='root path to source domain image folder')
parser.add_argument('--image_path_target', default='/extendspace/yizhou/DR/ODIR-5K', type=str, help='root path to target domain image folder')
parser.add_argument('--network', default='2D_unet', type=str, help='network architecture')
parser.add_argument('--global_pool', default='PCAM', type=str, help='global pooling method for CAM')
parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
parser.add_argument('--base_lr', default=0.01, type=float, help='base learning rate for SGD optimizer')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD / Adam')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD')
parser.add_argument('--ckpt_dir', default='./save_models/ALL', type=str, help='ckpt path')
parser.add_argument('--num_epochs', default=80, type=int, help='training epochs')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='Support one GPU & multiple GPUs')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes')
parser.add_argument('--class_idx', default=4, type=int, help='the index of class for training [0: ma, 1: ex, 2: se, 3: he, 4: all]')
parser.add_argument('--augmentation', default=True, type=bool, help='whether to do data augmentation')
parser.add_argument('--split_index', default=2, type=int, help='train_test_split index')
parser.add_argument('--save_path', default='./save_images', type=str, help='root path to save image folder')
args = parser.parse_args()

args.num_classes = [1, 1, 1, 1]


def val(data_loader, model):
    end = time.time()
    # model.train()
    model.eval()

    with torch.no_grad():
        for i, image in enumerate(data_loader):
            print(i)

            if args.gpu is not None:
                image = image.cuda()

            output, _, output_map = model(image)

            dict = {0: 'MA', 1: 'EX', 2: 'SE', 3: 'HE'}
            # save fine-level maps
            for t in range(args.n_classes):
                pred = torch.sigmoid(output[t])

                if t == 0:
                    pred_mask = (pred > 0.25).float()
                    save_sub_path = '/MA'
                    if not os.path.exists(args.save_path + '/Fine' + save_sub_path):
                        os.makedirs(args.save_path + '/Fine' + save_sub_path)
                elif t == 1:
                    pred_mask = (pred > 0.25).float()
                    save_sub_path = '/EX'
                    if not os.path.exists(args.save_path + '/Fine' + save_sub_path):
                        os.makedirs(args.save_path + '/Fine' + save_sub_path)
                elif t == 2:
                    pred_mask = (pred > 0.25).float()
                    save_sub_path = '/SE'
                    if not os.path.exists(args.save_path + '/Fine' + save_sub_path):
                        os.makedirs(args.save_path + '/Fine' + save_sub_path)
                elif t == 3:
                    pred_mask = (pred > 0.25).float()
                    save_sub_path = '/HE'
                    if not os.path.exists(args.save_path + '/Fine' + save_sub_path):
                        os.makedirs(args.save_path + '/Fine' + save_sub_path)

                image_save_fine(args.save_path + '/Fine' + save_sub_path, image, pred_mask, batch_idx=i, batch_size=image.size()[0])

            # save coarse-level maps
            for t in range(args.n_classes):
                prob_map = torch.sigmoid(output_map[t]).float()

                if t == 0:
                    save_sub_path = '/MA'
                    if not os.path.exists(args.save_path + '/Coarse' + save_sub_path):
                        os.makedirs(args.save_path + '/Coarse' + save_sub_path)
                elif t == 1:
                    save_sub_path = '/EX'
                    if not os.path.exists(args.save_path + '/Coarse' + save_sub_path):
                        os.makedirs(args.save_path + '/Coarse' + save_sub_path)
                elif t == 2:
                    save_sub_path = '/SE'
                    if not os.path.exists(args.save_path + '/Coarse' + save_sub_path):
                        os.makedirs(args.save_path + '/Coarse' + save_sub_path)
                elif t == 3:
                    save_sub_path = '/HE'
                    if not os.path.exists(args.save_path + '/Coarse' + save_sub_path):
                        os.makedirs(args.save_path + '/Coarse' + save_sub_path)

                image_save_coarse(args.save_path + '/Coarse' + save_sub_path, image, prob_map, batch_idx=i, batch_size=image.size()[0])

    val_time = time.time() - end

    return


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
    else:
        print('TODO')

    print('Loading Weights...')
    all_model_dir = os.path.join(args.ckpt_dir, 'epoch-208-dice-0.550-roc-0.843-pr-0.231.pth.tar')

    if args.class_idx == 4:
        model = load_model(model, all_model_dir)
    else:
        print('TODO')

    if args.gpu is not None:
        model = nn.DataParallel(model).cuda()

    lr = args.base_lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    optimizers = {
        'SGD': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'RMSprop': torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-8),
        'Adam': torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    }
    optimizer = optimizers[str(args.optimizer)]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30)

    print('Setting up data...')
    dataset_s, val_list = get_FGADR_Seg_test(image_path=args.image_path_source, network=args.network, aug=args.augmentation)

    val_loader = torch.utils.data.DataLoader(
        dataset_s['val'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    dict={0: 'Coarse', 1: 'Fine'}
    save_folder_coarse = os.path.join(args.save_path, '{}'.format(dict[0]))
    save_folder_fine = os.path.join(args.save_path, '{}'.format(dict[1]))
    if not os.path.exists(save_folder_coarse):
        os.makedirs(save_folder_coarse)
    if not os.path.exists(save_folder_fine):
        os.makedirs(save_folder_fine)

    print('Start Testing...')

    val(val_loader, model)

    print('Testing Finished.')


if __name__ == '__main__':
    main(args)
