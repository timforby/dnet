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
                self.img_files.append(file.readlines())
        if len(self.img_files) > 1:
            for l in self.img_files[1:]:
                assert len(self.img_files[0])==len(l), \
                    "Number of imagery images does not match number of other images"
        self.patch_shape = (patch_size, patch_size)
        self.data_length = len(self.img_files[0])
        self.image_index = -1
        self.patch_index = -1
        self.current_patches = []
        self.patch_lengths = []
        self.load_images()

    def load_images(self):
        #---------
        #  Image -- Imagery
        #---------

        self.image_index += 1
        imgs = []
        imgs_path = []
        for img_files in self.img_files:
            img_path = img_files[self.image_index % self.data_length].rstrip()
            img = cv2.imread(img_path)
            # Handles images with one channel
            if len(img.shape) == 2:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
            imgs.append(img)
            imgs_path.append(img_path)

        if len(imgs) > 1:
            for i,l in enumerate(imgs[1:]):
                assert imgs[0].shape==l.shape, \
                    "Imagery shape with "+str(imgs.shape) \
                    +" does not match other image with "+str(l.shape) \
                    +"\nPaths are "+imgs_path[0][self.image_index % self.data_length] +"\nand"\
                    +imgs_path[i][self.image_index % self.data_length]

        for img in imgs:
            mp = img.shape[0]*img.shape[1]//(self.patch_shape[0]*self.patch_shape[1])
            patches = patchify(img,self.patch_shape,max_patches=mp,random_state=0).transpose([0,3,1,2])
            self.current_patches.append(patches)
        self.patch_lengths.append(len(self.current_patches[0]))
        
    def __getitem__(self, index):
        self.patch_index +=1
        if self.patch_index >= self.patch_lengths[self.image_index]:
            self.load_images()
            self.patch_index = 0
        patch = [torch.from_numpy(patches[self.patch_index]).cuda().float() for patches in self.current_patches]
        return self.img_files[0][self.image_index], \
            self.patch_index, \
            self.patch_lengths[self.image_index], \
            patch
           

    def __len__(self):
        return len(self.img_files[0]*30)