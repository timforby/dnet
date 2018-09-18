import numpy as np
import cv2
import torch
import torchvision.utils as vutils


def show(x):
    y = np.array(torch.transpose(x,0,2))
    print(y.shape)
    cv2.imshow("sdjk",y)
    cv2.waitKey()

def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias = drange_out[0]-drange_in[0]*scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        ])
    return transform(x)
    
 
def save_image_single(img, path, imsize=512):
    vutils.save_image(img,'%s' % (path),normalize=False)
    
     
def save_image_grid(img, path, imsize=512, ngrid=4):
    vutils.save_image(img,'%s' % (path),normalize=False)