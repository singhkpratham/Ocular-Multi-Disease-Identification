import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
import glob
from PIL import Image
import random


class FGADR_Seg(Dataset):
    def __init__(self, data_dir, list, phase, transforms=None):
        super(FGADR_Seg, self).__init__()

        self.root_path = data_dir
        self.Image_files = glob.glob(os.path.join(data_dir, 'Original_Images/*'))
        self.MA_masks = glob.glob(os.path.join(data_dir, 'Microaneurysms_Masks/*'))
        self.EX_masks = glob.glob(os.path.join(data_dir, 'HardExudate_Masks/*'))
        self.SE_masks = glob.glob(os.path.join(data_dir, 'SoftExudate_Masks/*'))
        self.HE_masks = glob.glob(os.path.join(data_dir, 'Hemohedge_Masks/*'))

        self.Image_files_part = []
        self.MA_masks_part = []
        self.EX_masks_part = []
        self.SE_masks_part = []
        self.HE_masks_part = []

        [self.Image_files_part.append(self.Image_files[int(i)]) for i in list]
        [self.MA_masks_part.append(self.MA_masks[int(i)]) for i in list]
        [self.EX_masks_part.append(self.EX_masks[int(i)]) for i in list]
        [self.SE_masks_part.append(self.SE_masks[int(i)]) for i in list]
        [self.HE_masks_part.append(self.HE_masks[int(i)]) for i in list]

        self.transforms = transforms
        self.phase = phase
        print(len(self.Image_files_part))

    def __len__(self):
        return len(self.Image_files_part)

    @classmethod
    def preprocess(cls, pil_img, phase, x, y, h, w, p, angle):
        if phase == 'train':
            pil_img = TF.crop(pil_img, x, y, h, w)
            if p > 0.5:
                pil_img = TF.hflip(pil_img)
            pil_img = TF.rotate(pil_img, angle)

        if phase == 'val':
            pil_img = TF.crop(pil_img, x, y, h, w)

        img_nd = np.array(pil_img)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        if len(img_nd.shape) == 2:
            img_nd[img_nd > 0.5] = 1
            img_nd[img_nd < 0.5] = 0
            img_nd = np.expand_dims(img_nd, axis=2)

        # get classifcation label
        cls_label = np.zeros(1) # give a dummy value for input fundus image
        if img_nd.max() == 0:
            cls_label[0] = 0
        else:
            cls_label[0] = 1
        # print(cls_label)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans, cls_label

    def __getitem__(self, i):
        img = Image.open(self.Image_files_part[i]) #.resize((1280,1280))
        ma_mask = Image.open(self.MA_masks_part[i]) #.resize((1280,1280))
        ex_mask = Image.open(self.EX_masks_part[i]) #.resize((1280,1280))
        se_mask = Image.open(self.SE_masks_part[i]) #.resize((1280,1280))
        he_mask = Image.open(self.HE_masks_part[i]) #.resize((1280,1280))

        if len(np.array(ma_mask).shape) == 3:
            ma_mask = ma_mask.convert("L")
            ex_mask = ex_mask.convert("L")
            se_mask = se_mask.convert("L")
            he_mask = he_mask.convert("L")

        # print(np.array(ma_mask).shape)
        assert img.size[0:1] == ma_mask.size[0:1], \
            f'Image and mask should have the same spatial size'

        x, y, h, w = transforms.RandomCrop.get_params(
            img, output_size=(512, 512))
        p = random.random()
        angle = transforms.RandomRotation.get_params([-180, 180])

        # print((x, y, h, w, p, angle))

        img, img_label = self.preprocess(img, self.phase, x, y, h, w, p, angle)
        ma_mask, ma_label = self.preprocess(ma_mask, self.phase, x, y, h, w, p, angle)
        ex_mask, ex_label = self.preprocess(ex_mask, self.phase, x, y, h, w, p, angle)
        se_mask, se_label = self.preprocess(se_mask, self.phase, x, y, h, w, p, angle)
        he_mask, he_label = self.preprocess(he_mask, self.phase, x, y, h, w, p, angle)

        cls_labels = np.concatenate((ma_label, ex_label, se_label, he_label), axis=0)


        # print((img[0,:,:].mean(),img[1,:,:].mean(),img[2,:,:].mean()))
        """
        print(ma_mask.shape)
        arr_ma = ma_mask.flatten()  # 数组转为1维
        arr_ma = pd.Series(arr_ma)  # 转换数据类型
        arr_ma = arr_ma.value_counts()  # 计数
        arr_ma.sort_index(inplace=True)
        print(arr_ma)
        """

        return torch.from_numpy(img).float(), torch.from_numpy(ma_mask).float(), torch.from_numpy(ex_mask).float(), \
                torch.from_numpy(se_mask).float(), torch.from_numpy(he_mask).float(), torch.from_numpy(cls_labels).float()


class IRMA_NV(Dataset):
    def __init__(self, data_dir, list, phase, transforms=None):
        super(IRMA_NV, self).__init__()

        self.root_path = data_dir
        self.Image_files = glob.glob(os.path.join(data_dir, 'image/*'))
        self.masks = glob.glob(os.path.join(data_dir, 'mask/*'))

        self.Image_files_part = []
        self.masks_part = []

        [self.Image_files_part.append(self.Image_files[int(i)]) for i in list]
        [self.masks_part.append(self.masks[int(i)]) for i in list]

        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.Image_files_part)

    @classmethod
    def preprocess(cls, pil_img, phase, x, y, h, w, p, angle):
        if phase == 'train':
            pil_img = TF.crop(pil_img, x, y, h, w)
            if p > 0.5:
                pil_img = TF.hflip(pil_img)
            pil_img = TF.rotate(pil_img, angle)

        if phase == 'val':
            pil_img = TF.crop(pil_img, x, y, h, w)

        img_nd = np.array(pil_img)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        if len(img_nd.shape) == 2:
            img_nd[img_nd > 0.5] = 1
            img_nd[img_nd < 0.5] = 0
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, i):
        img = Image.open(self.Image_files_part[i]) #.resize((1280,1280))
        mask = Image.open(self.masks_part[i]) #.resize((1280,1280))

        if len(np.array(mask).shape) == 3:
            mask = mask.convert("L")

        assert img.size[0:1] == mask.size[0:1], \
            f'Image and mask should have the same spatial size'

        x, y, h, w = transforms.RandomCrop.get_params(
            img, output_size=(512, 512))
        p = random.random()
        angle = transforms.RandomRotation.get_params([-180, 180])

        img = self.preprocess(img, self.phase, x, y, h, w, p, angle)
        mask = self.preprocess(mask, self.phase, x, y, h, w, p, angle)

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


class FGADR_Seg_test(Dataset):
    def __init__(self, data_dir, list, phase, transforms=None):
        super(FGADR_Seg_test, self).__init__()

        self.root_path = data_dir
        self.Image_files = sorted(glob.glob(os.path.join(data_dir, '*')))

        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.Image_files)

    @classmethod
    def preprocess(cls, pil_img, phase, x, y, h, w, p, angle):
        if phase == 'train':
            pil_img = TF.crop(pil_img, x, y, h, w)
            if p > 0.5:
                pil_img = TF.hflip(pil_img)
            pil_img = TF.rotate(pil_img, angle)


        img_nd = np.array(pil_img)

        if img_nd.max() > 1:
            img_nd = img_nd / 255

        if len(img_nd.shape) == 2:
            img_nd[img_nd > 0.5] = 1
            img_nd[img_nd < 0.5] = 0
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, i):
        img = Image.open(self.Image_files[i]) #.resize((1280,1280))


        x, y, h, w = transforms.RandomCrop.get_params(
            img, output_size=(512, 512))
        p = random.random()
        angle = transforms.RandomRotation.get_params([-180, 180])

        # print((x, y, h, w, p, angle))

        img = self.preprocess(img, self.phase, x, y, h, w, p, angle)


        return torch.from_numpy(img).float()


class ODIR(Dataset):
    def __init__(self, image_list, label, phase, transform=None):
        super(ODIR, self).__init__()

        self.Image_files = image_list
        self.Label_files = label
        self.transforms = transform
        self.phase = phase
        print(len(self.Image_files))

    def __len__(self):
        return len(self.Image_files)

    @classmethod
    def preprocess(cls, pil_img, phase, x, y, h, w, p, angle):
        if phase == 'train':
            pil_img = TF.crop(pil_img, x, y, h, w)
            if p > 0.5:
                pil_img = TF.rotate(pil_img, angle)

        img_nd = np.array(pil_img)

        img_nd = img_nd / 255

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, i):
        img = Image.open(self.Image_files[i])
        # img_ori_nd = np.array(img)
        # print(img_ori_nd.max())
        # print(img_ori_nd.min())
        grade = np.zeros(8)
        j = 0
        for label in self.Label_files[i]:
            grade[j] = int(label)
            j += 1
        # print(self.Image_files[i])

        resize = transforms.Resize(size=(640, 640))
        img = resize(img)

        x, y, h, w = transforms.RandomCrop.get_params(
            img, output_size=(512, 512))
        p = random.random()
        angle = transforms.RandomRotation.get_params([-180, 180])

        img_trans = self.preprocess(img, self.phase, x, y, h, w, p, angle)
        # if self.transforms:
            # img = self.transforms(img)

        # img_nd = np.array(img)
        # print(img_nd.max())
        # print(img_nd.min())

        return torch.from_numpy(img_trans).float(), torch.from_numpy(grade).float()