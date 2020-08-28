# this function is used to concat different data files
# train and test splits

import os
import glob
from .dataloader import FGADR_Seg, FGADR_Seg_test, ODIR
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import csv

def get_FGADR_Seg(image_path, split_index, network, batch_size, aug=False):
    if not aug:
        transform = None
    else:
        transform = get_transform()

    data_dir = image_path
    train_csv = os.path.join(data_dir, 'train_%02d.csv' % split_index)
    test_csv = os.path.join(data_dir, 'test_%02d.csv' % split_index)

    train_list = []
    with open(train_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            train_list.append(row[0])
        csvfile.close()

    len_train_list = len(train_list)
    rem_train = len_train_list % batch_size
    # print(train_list[0:(batch_size - rem_train)])
    train_list_pad = train_list + train_list[0:(batch_size - rem_train)]

    val_list = []
    with open(test_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            val_list.append(row[0])
        csvfile.close()

    len_val_list = len(val_list)
    rem_val = len_val_list % batch_size
    # print(val_list[0:(batch_size - rem_val)])
    val_list_pad = val_list + val_list[0:(batch_size - rem_val)]

    if network == 'dense_unet':
        print('processing 2d format data...')
        # TODO
        datasets = {}
        datasets['train'] = FGADR_Seg(data_dir, list=train_list_pad, phase='train', transforms=transform['train'])
        datasets['val'] = FGADR_Seg(data_dir, list=val_list_pad, phase='val', transforms=transform['val'])
    else:
        print('TODO')
    return datasets


def get_FGADR_Seg_test(image_path, network, aug=False):
    if not aug:
        transform = None
    else:
        transform = get_transform()

    val_list = sorted(glob.glob(os.path.join(image_path, '*')))

    datasets = {}
    datasets['val'] = FGADR_Seg_test(image_path, list=val_list, phase='val', transforms=transform['val'])

    return datasets, val_list


def get_ODIR(dataset_path, input_size, batch_size, split_index):

    dataset_csv = os.path.join(dataset_path, 'label_v3.csv')
    image_path_list = []
    image_label_list = []
    with open(dataset_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = next(csv_reader)
        for row in csv_reader:
            image_path = os.path.join(dataset_path + '/images', row[0]+'.jpg')
            image_path_list.append(image_path)
            image_label_list.append(row[1:])
        csvfile.close()

    data_dir = (dataset_path)
    train_csv = os.path.join(data_dir, 'train_%02d.csv' % split_index)
    test_csv = os.path.join(data_dir, 'test_%02d.csv' % split_index)

    train_image_list = []
    train_y_list = []
    with open(train_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            train_image_list.append(image_path_list[int(row[0])])
            train_y_list.append(image_label_list[int(row[0])])
        csvfile.close()

    val_image_list = []
    val_y_list = []
    with open(test_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            val_image_list.append(image_path_list[int(row[0])])
            val_y_list.append(image_label_list[int(row[0])])
        csvfile.close()

    # val_size = 0.2
    # train_image_list, val_image_list, train_y_list, val_y_list  = \
    #     train_test_split(image_path_list, image_label_list, test_size=val_size)

    len_train_list = len(train_image_list)
    rem_train = len_train_list % batch_size
    # print(train_image_list[0:(batch_size - rem_train)])
    # print(train_y_list[0:(batch_size - rem_train)])
    train_image_list_pad = train_image_list + train_image_list[0:(batch_size - rem_train)]
    train_y_list_pad = train_y_list + train_y_list[0:(batch_size - rem_train)]

    len_val_list = len(val_image_list)
    rem_val = len_val_list % batch_size
    # print(val_image_list[0:(batch_size - rem_val)])
    # print(val_y_list[0:(batch_size - rem_val)])
    val_image_list_pad = val_image_list + val_image_list[0:(batch_size - rem_val)]
    val_y_list_pad = val_y_list + val_y_list[0:(batch_size - rem_val)]

    transform = {}
    transform['train'] = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform['test'] = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    datasets = {}
    datasets['train'] = ODIR(image_list=train_image_list_pad, label=train_y_list_pad, phase='train', transform=transform['train'])
    datasets['val'] = ODIR(image_list=val_image_list_pad, label=val_y_list_pad, phase='test', transform=transform['test'])

    # compute weights of different classes
    dict = {}
    for row_label in image_label_list:
        res = [idx for idx, value in enumerate(row_label) if int(value) == 1]
        for key in res:
            dict[key] = dict.get(key, 0) + 1
    print(dict)

    weights = np.zeros(8)
    weights[0] = 1 / (dict[0] / len(image_label_list))
    weights[1] = 1 / (dict[1] / len(image_label_list))
    weights[2] = 1 / (dict[2] / len(image_label_list))
    weights[3] = 1 / (dict[3] / len(image_label_list))
    weights[4] = 1 / (dict[4] / len(image_label_list))
    weights[5] = 1 / (dict[5] / len(image_label_list))
    weights[6] = 1 / (dict[6] / len(image_label_list))
    weights[7] = 1 / (dict[7] / len(image_label_list))

    weights = weights / sum(weights) * [1, 1, 1, 1, 1, 1, 1, 1]
    print(weights)

    return datasets['train'], datasets['val'], torch.from_numpy(weights).float()


def get_transform():
    # Now, only do the normalization

    mean = ([0.5, 0.5, 0.5])
    std = ([0.5, 0.5, 0.5])

    transform = {}
    transform['train'] = transforms.Compose([
        # transforms.RandomCrop(1000),
        # transforms.Resize(512),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform['val'] = transforms.Compose([
        # transforms.RandomCrop(1000),
        # transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform

