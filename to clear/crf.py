import os
import sys
import numpy as np
import argparse
from scipy import misc
import load,proc,densecrf
import classes as cl

ar = argparse.ArgumentParser()
ar.add_argument('path1', type = str, help = "Path to label images")
args = ar.parse_args()
path1 = args.path1


path1 =  path1+'/'
path2 =  '../../data/real/'
result = path1+'crf/'
if not os.path.exists(result):
    os.makedirs(result)
labels = load.load_data_simp([path1])  
set_imgs = load.load_all(path2,comp=True)
x_img = set_imgs[1]
x_img = proc._join(x_img,set_imgs[2:])
    
for i in range(len(x_img)):
    x_i = x_img[i]
    label = labels[0][i]
    crflabel = densecrf.crf(label,x_i,classes=2,ls=[0,3])
    crflabel = cl.declassimg2(crflabel,map=True)
    number = str(i).zfill(4)
    misc.imsave(result+number+".png",crflabel)
