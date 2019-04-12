import os
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from sklearn.feature_extraction.image import extract_patches_2d as patchify
from skimage.transform import resize

import sys

class ListDataset(Dataset):
    def __init__(self, list_paths, patch_size=400):
        self.img_files = []
        for list_path in list_paths:
            with open(list_path, 'r') as file:
                self.img_files.append([l.split(',') for l in file.readlines()])
        for l in self.img_files[1:]:
            assert len(self.img_files[0])==len(l), \
                "Number of imagery images does not match number of other images"
        self.patch_shape = (patch_size, patch_size)
        self.data_length = -1
        self.image_index = -1
        self.current_patches = []
        self.patch_lengths = [self.get_num_patch(int(x),int(y)) for _,x,y,_ in self.img_files[0]]

    def get_num_patch(self,x,y):
        return 4*x*y//(self.patch_shape[0]*self.patch_shape[1])

    def load_images(self):
        #---------
        #  Image -- Imagery
        #---------
        imgs = []
        imgs_path = []
        for img_files in self.img_files:
            img_path = img_files[self.image_index % self.data_length][0].rstrip()
            img = cv2.imread(img_path)
            # Handles images with one channel
            if len(img.shape) == 2:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
            imgs.append(img)
            imgs_path.append(img_path)

        for i,l in enumerate(imgs[1:]):
            assert imgs[0].shape==l.shape, \
                "Imagery shape with "+str(imgs.shape) \
                +" does not match other image with "+str(l.shape) \
                +"\nPaths are "+imgs_path[0][self.image_index % self.data_length] +"\nand"\
                +imgs_path[i][self.image_index % self.data_length]

        self.current_patches = []
        for img in imgs:
            mp = self.patch_lengths[self.image_index]
            self.current_patches.append(patchify(img,self.patch_shape,max_patches=mp,random_state=0).transpose([0,3,1,2]))
        
    def __getitem__(self, index):
        patch_index = index
        tmp_img_idx = 0
        while patch_index%self.__len__() >= self.patch_lengths[tmp_img_idx]:
            patch_index -=self.patch_lengths[tmp_img_idx]
            tmp_img_idx = (tmp_img_idx+1)%len(self.img_files[0])
        if tmp_img_idx != self.image_index:
            self.image_index = tmp_img_idx
            self.load_images()
        patch = [torch.from_numpy(patches[patch_index]).float() for patches in self.current_patches]
        return self.img_files[0][self.image_index], \
            self.image_index, \
            len(self.img_files[0]), \
            patch
           

    def __len__(self):
        if self.data_length == -1:
            self.data_length = sum(self.patch_lengths)

        return self.data_length