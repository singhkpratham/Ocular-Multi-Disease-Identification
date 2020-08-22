import torch
import torch.nn as nn
import numpy as np
import cv2
import os

def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def image_save_fine(save_path, image, pred, batch_idx, batch_size):

    save_img = image.cpu().detach().numpy()
    save_mask = pred.cpu().detach().numpy()
    # save_gt = gt.cpu().detach().numpy()

    for i in range(batch_size):
        save_base_name = '{}_{}_'.format(batch_idx, i)
        save_img_name = os.path.join(save_path, save_base_name + 'img.jpg')
        save_mask_name = os.path.join(save_path, save_base_name + 'mask.jpg')
        # save_gt_name = os.path.join(save_path, save_base_name + 'gt.jpg')
        cv2.imwrite(save_img_name, cv2.cvtColor(save_img.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0), cv2.COLOR_RGB2BGR))
        cv2.imwrite(save_mask_name, save_mask.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0))
        # cv2.imwrite(save_gt_name, save_gt.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0))


def image_save_coarse(save_path, image, prob_map, batch_idx, batch_size):

    save_img = image.cpu().detach().numpy()
    ori_size_H, ori_size_W = image.size()[2], image.size()[3]
    # print(prob_map.size())
    save_mask = prob_map.cpu().detach().numpy().squeeze()
    # save_gt = gt.cpu().detach().numpy()

    for i in range(batch_size):
        save_base_name = '{}_{}_'.format(batch_idx, i)
        # save_base_name = os.path.basename(batch_list[i])[:-4]
        save_img_name = os.path.join(save_path, save_base_name + '_img.jpg')
        save_mask_name = os.path.join(save_path, save_base_name + '_mask.jpg')
        # save_gt_name = os.path.join(save_path, save_base_name + 'gt.jpg')
        cv2.imwrite(save_img_name, cv2.cvtColor(save_img.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0), cv2.COLOR_RGB2BGR))
        save_map = (save_mask[i, :, :].squeeze()*255.0).astype(np.uint8)
        cv2.imwrite(save_mask_name, cv2.applyColorMap(cv2.resize(save_map, (ori_size_H, ori_size_W)), cv2.COLORMAP_JET))
        # cv2.imwrite(save_gt_name, save_gt.transpose((0, 2, 3, 1))[i, :, :, :].squeeze()*(255.0))
