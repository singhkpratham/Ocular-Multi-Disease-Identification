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
from torch.autograd import Variable
from sklearn.metrics import auc
from itertools import cycle


from networks import DenseUNet, UNet2D, DomainSpecificDiscriminator
from datasets.get_dataset import get_FGADR_Seg, get_ODIR
from utils.Layers import *
from utils.Losses import *
from utils.seg_eval_metrics import *
from utils.utils import *
from utils import Logger, AverageMeter, odir_metrics, mkdir_p, savefig


parser = argparse.ArgumentParser(description='Inductive Transfer Learning - Multi-scale Transfer Connections & Domain-Specific Adversarial Learning.')

parser.add_argument('--image_path_source', default='/extendspace/yizhou/DR/FGADR-2842/Segmentation', type=str, help='root path to source domain image folder')
parser.add_argument('--image_path_target', default='/extendspace/yizhou/DR/ODIR-5K', type=str, help='root path to target domain image folder')
parser.add_argument('--input_size', default=512, type=int, help='the input size of images')
parser.add_argument('--network', default='dense_unet', type=str, help='network architecture: 2D_unet, dense_unet')
parser.add_argument('--global_pool', default='PCAM', type=str, help='global pooling method for CAM')
parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
parser.add_argument('--base_lr', default=0.001, type=float, help='base learning rate for SGD optimizer')
parser.add_argument('--optimizer', default='Adam', type=str, help='SGD / Adam')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD')
parser.add_argument('--ckpt_dir', default='./save_models', type=str, help='ckpt path')
parser.add_argument('--pretrain', default='pretrained/epoch-97-pr-0.553.pth.tar', type=str, help='path to the pretrained model')
parser.add_argument('--num_epochs', default=300, type=int, help='training epochs')
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str, help='Support one GPU & multiple GPUs')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes of source domain')
parser.add_argument('--t_classes', default=8, type=int, help='number of classes of target domain')
parser.add_argument('--class_idx', default=4, type=int, help='the index of class for training [0: ma, 1: ex, 2: se, 3: he, 4: all, 5: irma, 6: nv]')
parser.add_argument('--augmentation', default=True, type=bool, help='whether to do data augmentation')
parser.add_argument('--split_index', default=1, type=int, help='train_test_split index')
args = parser.parse_args()

args.num_classes = [1, 1, 1, 1]


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def get_cls_loss(output_cls, gt_cls, index, args):
    for num_class in args.num_classes:
        assert num_class == 1
    gt_cls = gt_cls[:, index].view(-1)
    # print('The number of one: %d' % np.count_nonzero(gt_cls.cpu() == 1))
    loss_cls = F.binary_cross_entropy_with_logits(
        output_cls[index].view(-1), gt_cls)

    label = torch.sigmoid(output_cls[index].view(-1)).ge(0.5).float()
    acc = (gt_cls == label).float().sum() / len(label)

    return (loss_cls, acc)


def train(data_loader_s, data_loader_t, class_weights_t, model, discriminator, epoch, optimizer, optimizer_D, args):
    end = time.time()
    model.train()

    # Metric values initiation
    metrics = []
    print('Length of source domain dataloader:', len(data_loader_s))
    print('Length of target domain dataloader:', len(data_loader_t))
    n_train = max(len(data_loader_s), len(data_loader_t))
    pos_count_batch = [0, 0, 0, 0]
    dice_total = [0, 0, 0, 0]
    auc_roc_total = [0, 0, 0, 0]
    auc_pr_total = [0, 0, 0, 0]

    losses_cls = AverageMeter()
    score_kappa = AverageMeter()
    score_auc = AverageMeter()
    score_f1 = AverageMeter()
    acc_cls = np.zeros(args.t_classes)

    for param_group in optimizer.param_groups:
        print('Epoch %d (lr %.5f)' % (epoch, param_group['lr']))


    for i, data_tuple in enumerate(zip(cycle(data_loader_s), data_loader_t)):

        image, ma_mask, ex_mask, se_mask, he_mask, _ = data_tuple[0]
        inputs, targets = data_tuple[1]

        if args.gpu is not None:
            image, ma_mask, ex_mask, se_mask, he_mask = image.cuda(), ma_mask.cuda(), ex_mask.cuda(), se_mask.cuda(), he_mask.cuda()
            inputs, targets = inputs.cuda(), targets.cuda()

        gt_mask = list()
        gt_mask.append(ma_mask)
        gt_mask.append(ex_mask)
        gt_mask.append(se_mask)
        gt_mask.append(he_mask)

        output, output_t, feat_s, feat_t = model(image, inputs)
        # print(feat_s.size())
        # print(feat_t.size())

        # Source domain optimization
        loss_seg = 0
        pos_count = [0, 0, 0, 0]
        for t in range(args.n_classes):

            gt = gt_mask[t]
            # compute weights for batch elements
            batch_weights = np.zeros((image.size()[0], 1, 512, 512))
            for j in range(image.size()[0]):
                if torch.max(gt[j, :, :, :]) == 0:
                    batch_weights[j, :, :, :] = 0
                else:
                    batch_weights[j, :, :, :] = 1
                    pos_count[t] += 1
            # print(batch_weights)

            if pos_count[t] > 0:
                pos_count_batch[t] += 1

            criterion_1 = BCELoss(n_classes=1, batch_weights=batch_weights)
            criterion_2 = DiceCoeff()

            output_s = output[:, t, :, :].unsqueeze(dim=1)
            loss_1 = criterion_1(output_s, gt)
            loss_2 = criterion_2(output_s, gt)

            loss_seg += loss_1 + loss_2*0.2

        # Target domain optimization
        criterion = nn.BCEWithLogitsLoss(weight=class_weights_t.cuda())
        loss_cls = criterion(output_t, targets)


        # Adversarial ground truths and criterion
        valid = Variable(torch.cuda.FloatTensor(inputs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(image.shape[0], 1).fill_(0.0), requires_grad=False)

        criterion_adv = torch.nn.BCELoss()

        loss_g = criterion_adv(discriminator(feat_s, domain_label=0), valid)

        # Task loss + G loss
        loss_output = loss_seg + loss_cls + 0.5 * loss_g

        optimizer.zero_grad()
        # loss_output.backward()
        loss_output.backward(retain_graph=True)
        optimizer.step()

        # Measure discriminator's ability to classify target domain data from source domain data
        loss_real = criterion_adv(discriminator(feat_t, domain_label=1), valid)
        loss_fake = criterion_adv(discriminator(feat_s, domain_label=0), fake)
        loss_d = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_d.backward()
        optimizer_D.step()


        # Evaluation during training - source domain
        loss_seg = loss_seg.item()
        metrics.append([loss_seg, loss_seg])

        for t in range(args.n_classes):
            gt = gt_mask[t]
            output_s = output[:, t, :, :].unsqueeze(dim=1)
            pred = torch.sigmoid(output_s)
            _, auc_roc_batch, auc_pr_batch = seg_metrics(pred, gt, pos_count[t])
            auc_roc_total[t] += auc_roc_batch
            auc_pr_total[t] += auc_pr_batch
            pred = (pred > 0.5).float()
            """
            print(pred.shape)
            arr_pred = pred.cpu().flatten()
            arr_pred = pd.Series(arr_pred)
            arr_pred = arr_pred.value_counts()
            arr_pred.sort_index(inplace=True)
            print(arr_pred)
            """
            dice_total[t] += dice_coeff(pred, gt).item()

        # Evaluation during training - target domain
        predicts = nn.Sigmoid()(output_t)
        kappa, auc, f1, final, acc_each_label = odir_metrics(predicts.data, targets.data)
        acc_cls += acc_each_label
        losses_cls.update(loss_cls.data, inputs.size(0))
        score_kappa.update(kappa)
        score_auc.update(auc)
        score_f1.update(f1)

        batch_time = time.time() - end
        end = time.time()

        if i % 10 == 0:
            print('Epoch {} [{}/{}] \t Segmentation Loss: {:.5f} \t Classification Loss: {:.5f} \t G Loss: {:.5f} \t D Loss: {:.5f}'.format(
                epoch, i, n_train, metrics[i][0], losses_cls.avg, loss_g.item(), loss_d.item()))

    metrics = np.asarray(metrics, np.float32)
    loss_epoch_mean = metrics.mean(axis=0)

    dice_total = np.asarray(dice_total, np.float32)
    dice_epoch_all = dice_total / n_train
    print('Dice Coeff: MA: %3.5f, EX: %3.5f, SE: %3.5f, HE: %3.5f' %(dice_epoch_all[0], dice_epoch_all[1], dice_epoch_all[2], dice_epoch_all[3]))
    dice_total_mean = dice_total.mean(axis=0)
    dice_epoch_avg = dice_total_mean / n_train

    auc_roc_epoch = [a / b for a, b in zip(auc_roc_total, pos_count_batch)]
    auc_roc_epoch = np.asarray(auc_roc_epoch, np.float32)
    print('AUC ROC: MA: %3.5f, EX: %3.5f, SE: %3.5f, HE: %3.5f' % (auc_roc_epoch[0], auc_roc_epoch[1], auc_roc_epoch[2], auc_roc_epoch[3]))
    auc_roc_epoch_avg = auc_roc_epoch.mean(axis=0)

    auc_pr_epoch = [a / b for a, b in zip(auc_pr_total, pos_count_batch)]
    auc_pr_epoch = np.asarray(auc_pr_epoch, np.float32)
    print('AUC PR: MA: %3.5f, EX: %3.5f, SE: %3.5f, HE: %3.5f' % (auc_pr_epoch[0], auc_pr_epoch[1], auc_pr_epoch[2], auc_pr_epoch[3]))
    auc_pr_epoch_avg = auc_pr_epoch.mean(axis=0)

    print('Epoch %d: Training Source Domain: Seg Average Loss: %2.5f, Dice Coeff: %3.5f, AUC_ROC: %3.5f, AUC_PR: %3.5f' %(epoch,
                        loss_epoch_mean[1], dice_epoch_avg, auc_roc_epoch_avg, auc_pr_epoch_avg))
    print('Epoch %d: Training Target Domain: Cls Average Loss: %2.5f, Kappa: %3.5f, AUC: %3.5f, F1: %3.5f' % (epoch, losses_cls.avg,
                        score_kappa.avg, score_auc.avg, score_f1.avg))

    acc_cls = acc_cls / n_train
    print(acc_cls)
    return loss_epoch_mean, losses_cls.avg


def val(data_loader_s, data_loader_t, class_weights_t, model, epoch):
    end = time.time()

    # Metric values initiation
    metrics = []
    n_val = max(len(data_loader_s), len(data_loader_t))
    pos_count_batch = [0, 0, 0, 0]
    dice_total = [0, 0, 0, 0]
    auc_roc_total = [0, 0, 0, 0]
    auc_pr_total = [0, 0, 0, 0]

    losses_cls = AverageMeter()
    score_kappa = AverageMeter()
    score_auc = AverageMeter()
    score_f1 = AverageMeter()
    acc_cls = np.zeros(args.t_classes)


    with torch.no_grad():
        for i, data_tuple in enumerate(zip(cycle(data_loader_s), data_loader_t)):

            image, ma_mask, ex_mask, se_mask, he_mask, _ = data_tuple[0]
            inputs, targets = data_tuple[1]

            if args.gpu is not None:
                image, ma_mask, ex_mask, se_mask, he_mask = image.cuda(), ma_mask.cuda(), ex_mask.cuda(), se_mask.cuda(), he_mask.cuda()
                inputs, targets = inputs.cuda(), targets.cuda()

            gt_mask = list()
            gt_mask.append(ma_mask)
            gt_mask.append(ex_mask)
            gt_mask.append(se_mask)
            gt_mask.append(he_mask)

            output, output_t, _, _ = model(image, inputs)

            # Source domain loss computation
            loss_seg = 0
            pos_count = [0, 0, 0, 0]
            for t in range(args.n_classes):

                gt = gt_mask[t]
                # compute weights for batch elements
                batch_weights = np.zeros((image.size()[0], 1, 512, 512))
                for j in range(image.size()[0]):
                    if torch.max(gt[j, :, :, :]) == 0:
                        batch_weights[j, :, :, :] = 0
                    else:
                        batch_weights[j, :, :, :] = 1
                        pos_count[t] += 1
                # print(batch_weights)

                if pos_count[t] > 0:
                    pos_count_batch[t] += 1

                criterion_1 = BCELoss(n_classes=1, batch_weights=batch_weights)
                criterion_2 = DiceCoeff()

                output_s = output[:, t, :, :].unsqueeze(dim=1)
                loss_1 = criterion_1(output_s, gt)
                loss_2 = criterion_2(output_s, gt)

                loss_seg += loss_1 + loss_2 * 0.2

            # Target domain optimization
            criterion = nn.BCEWithLogitsLoss(weight=class_weights_t.cuda())
            loss_cls = criterion(output_t, targets)

            # Evaluation during testing - source domain
            loss_seg = loss_seg.item()
            metrics.append([loss_seg, loss_seg])

            for t in range(args.n_classes):
                gt = gt_mask[t]
                output_s = output[:, t, :, :].unsqueeze(dim=1)
                pred = torch.sigmoid(output_s)
                _, auc_roc_batch, auc_pr_batch = seg_metrics(pred, gt, pos_count[t])
                auc_roc_total[t] += auc_roc_batch
                auc_pr_total[t] += auc_pr_batch
                pred = (pred > 0.5).float()
                dice_total[t] += dice_coeff(pred, gt).item()

            # Evaluation during testing - target domain
            predicts = nn.Sigmoid()(output_t)
            kappa, auc, f1, final, acc_each_label = odir_metrics(predicts.data, targets.data)
            acc_cls += acc_each_label
            losses_cls.update(loss_cls.data, inputs.size(0))
            score_kappa.update(kappa)
            score_auc.update(auc)
            score_f1.update(f1)

    val_time = time.time() - end

    metrics = np.asarray(metrics, np.float32)
    loss_epoch_mean = metrics.mean(axis=0)

    dice_total = np.asarray(dice_total, np.float32)
    dice_total_mean = dice_total.mean(axis=0)
    dice_epoch_avg = dice_total_mean / n_val

    auc_roc_epoch = [a / b for a, b in zip(auc_roc_total, pos_count_batch)]
    auc_roc_epoch = np.asarray(auc_roc_epoch, np.float32)
    auc_roc_epoch_avg = auc_roc_epoch.mean(axis=0)

    auc_pr_epoch = [a / b for a, b in zip(auc_pr_total, pos_count_batch)]
    auc_pr_epoch = np.asarray(auc_pr_epoch, np.float32)
    auc_pr_epoch_avg = auc_pr_epoch.mean(axis=0)

    print('Epoch %d: Validation Source Domain: Seg Average Loss: %2.5f, Dice Coeff: %3.5f, AUC_ROC: %3.5f, AUC_PR: %3.5f' %(epoch,
                        loss_epoch_mean[1], dice_epoch_avg, auc_roc_epoch_avg, auc_pr_epoch_avg))
    print('Epoch %d: Validation Target Domain: Cls Average Loss: %2.5f, Kappa: %3.5f, AUC: %3.5f, F1: %3.5f' % (epoch, losses_cls.avg,
                        score_kappa.avg, score_auc.avg, score_f1.avg))

    acc_cls = acc_cls / n_val
    print(acc_cls)
    return dice_epoch_avg, auc_roc_epoch_avg, auc_pr_epoch_avg, score_kappa.avg, score_auc.avg, score_f1.avg


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
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
        model = UNet2D(n_channels=3, n_classes=args.n_classes, num_classes=args.num_classes, pool=args.global_pool, bilinear=True)
    elif args.network == 'dense_unet':
        model = DenseUNet(n_channels=3, n_classes=args.n_classes, num_classes=args.num_classes,
                          t_classes=args.t_classes, pool=args.global_pool, bilinear=True)
    else:
        print('TODO')

    discriminator = DomainSpecificDiscriminator()

    # model.apply(weights_init)

    if args.pretrain:
        print('Loading pre-trained layers ...')
        pretrained = os.path.join(args.ckpt_dir, args.pretrain)
        model = load_model(model, pretrained)

    if args.gpu is not None:
        model = nn.DataParallel(model).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()

    lr = args.base_lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    optimizers = {
        'SGD': torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'RMSprop': torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-8),
        'Adam': torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0)
    }
    optimizer = optimizers[str(args.optimizer)]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.33, mode='min', patience=50)

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    print('Setting up source domain data...')
    dataset_s = get_FGADR_Seg(image_path=args.image_path_source, split_index=args.split_index, network=args.network,
                              batch_size=args.batch_size,  aug=args.augmentation)
    print('Setting up target domain data...')
    trainset_t, valset_t, class_weights_t = get_ODIR(dataset_path=args.image_path_target, input_size=args.input_size,
                                                     batch_size=args.batch_size, split_index=args.split_index)

    train_loader_s = data.DataLoader(dataset_s['train'], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader_s = data.DataLoader(dataset_s['val'], batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_loader_t = data.DataLoader(trainset_t, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader_t = data.DataLoader(valset_t, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    dict={0: 'MA', 1: 'EX', 2: 'SE', 3: 'HE', 4: 'ALL'}
    save_folder = os.path.join(args.ckpt_dir, '{}'.format(dict[args.class_idx]))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print('Start training...')
    start_epoch = 0
    best_val_score = 0

    if args.class_idx == 0:
        print('Train MA mask.')
    elif args.class_idx == 1:
        print('Train EX mask.')
    elif args.class_idx == 2:
        print('Train SE mask.')
    elif args.class_idx == 3:
        print('Train HE mask.')
    elif args.class_idx == 4:
        print('Train ALL masks.')

    for epoch in range(start_epoch, args.num_epochs):
        loss_epoch, losses_cls = train(train_loader_s, train_loader_t, class_weights_t, model, discriminator, epoch, optimizer, optimizer_D, args)
        val_dice_score, val_roc_score, val_pr_score, val_kappa, val_auc, val_f1 = val(val_loader_s, val_loader_t, class_weights_t, model, epoch)

        scheduler.step(val_kappa)
        if val_kappa > best_val_score:
            best_val_score = val_kappa
            path = '{}/epoch-{}-pr-{:.3f}-kappa-{:.3f}.pth.tar'.format(save_folder, epoch, val_pr_score, val_kappa)
            save_model(path, epoch, model, optimizer)
        if epoch == args.num_epochs-1:
            path = '{}/epoch-{}-pr-{:.3f}-kappa-{:.3f}.pth.tar'.format(save_folder, epoch, val_pr_score, val_kappa)
            save_model(path, epoch, model, optimizer)
    print('Training Finished.')

if __name__ == '__main__':
    main(args)
