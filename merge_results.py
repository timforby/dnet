import os
import sys
import numpy as np
import argparse
from scipy import misc
import load

ar = argparse.ArgumentParser()
ar.add_argument('path1', type = str, help = "Path to base images")
ar.add_argument('path2', type = str, help = "Path to images to copy into base")
ar.add_argument('-l','--labels', action='append', help="Option to select specific training labels \nOTHER = 0\nTREE = 1\nBUILDING = 2\nCAR = 3\nVEG = 4\nGROUND = 5")
args = ar.parse_args()
path1 = args.path1
path2 = args.path2
labels = args.labels
labels = list(map(int, labels))

path1 =  path1+'/'
path2 =  path2+'/'
result = path1+'result/'
if not os.path.exists(result):
    os.makedirs(result)
    
imgs_base = load.load_data_simp([path1,path2])
def to_color(res):
    if res == 0:
        return [1,0,0]
    elif res == 1:
        return [0,1,0]
    elif res == 2:
        return [0,0,1]
    elif res == 3:
        return [1,1,0]
    elif res == 4:
        return [0,1,1]
    elif res == 5:
        return [1,1,1]


for i in range(len(imgs_base[0])):
    img = imgs_base[1][i]
    bse = imgs_base[0][i]
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]
    for l in labels:
        color = to_color(l)
        msk = (r==color[0])&(g==color[1])&(b==color[2])
        bse[:,:,:3][msk] = color
    number = str(i).zfill(4)
    misc.imsave(result+number+".png",bse)
    print('done with: '+str(i))
