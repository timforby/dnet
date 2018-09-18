import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data

class runtimeloader(data.Dataset):
    def __init__(self, path_x, path_y, joint_transform=None, transform=None, target_transform=None):
        #where imgs will just be img.shape
        self.imgs = asses_dataset(path_x,path_y)
        self.img_path,self.mask_path = path_x,path_y
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        if len(self.imgs) == -1:
            raise RuntimeError('Error loading images, some assertion fail.')
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.sliding_crop is not None:
            img_slices, mask_slices, slices_info = self.sliding_crop(img, mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.target_transform(e) for e in mask_slices]
            img, mask = torch.stack(img_slices, 0), torch.stack(mask_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img, mask

    def __len__(self):
        return len(self.imgs)