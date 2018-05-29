import os
import sys
import numpy as np
import argparse
from scipy import misc

ar = argparse.ArgumentParser()
ar.add_argument('gpu', type = int, help = "Please enter GPU to use")
ar.add_argument('name', type = str, help = "Please model name")
ar.add_argument('patch_size', type = int, help = "Please enter model patch size")

ar.add_argument('--comp', action='store_true', help='Option defining whether process competition images')
args = ar.parse_args()
device = 'cuda'+str(args.gpu)
name = args.name

patch_size = (args.patch_size,args.patch_size)
comp = args.comp

#----Image Paths---

BASE_PATH = '../../data/'
REAL_PATH = BASE_PATH+'real/'
OUT_PATH = BASE_PATH+'out/'
     
TRAIN_PATH = OUT_PATH+name
if not os.path.exists(TRAIN_PATH):
    print("Model does not exist")
    sys.exit(0)

append = '_test' if not comp else '_test_comp'
RESULT_PATH_BASE = TRAIN_PATH+append

if not os.path.exists(RESULT_PATH_BASE):
    os.makedirs(RESULT_PATH_BASE)
RESULT_PATH_BASE = RESULT_PATH_BASE + '/edge'
os.makedirs(RESULT_PATH_BASE)

#----KERAS ENV-------

os.environ["THEANO_FLAGS"]='device='+device
sys.setrecursionlimit(50000)


import load,proc,classes,math
from keras.models import Model, load_model


def sp_argmax(x):
    if x[0]==-1:
        return 0
    else:
        return np.argmax(x)+1
    
def edgeim_to_patch(x_i, ps):
    patches = []
    for p in range(0,x_i.shape[0]//2,ps[0]):#horizleft
        patches.append(x_i[p:p+ps[0],0:ps[1],:])
        patches.append(x_i[p:p+ps[0],-ps[1]:,:])
      
    for p in range(x_i.shape[0],x_i.shape[0]//2,-ps[0]):#horizright
        patches.append(x_i[p-ps[0]:p,0:ps[1],:])
        patches.append(x_i[p-ps[0]:p,-ps[1]:,:])
   
    for p in range(ps[1],x_i.shape[1]//2,ps[1]):#vertup
        patches.append(x_i[0:ps[0],p:p+ps[1],:])
        patches.append(x_i[-ps[0]:,p:p+ps[1],:])

    for p in range(x_i.shape[1]-ps[1],x_i.shape[1]//2,-ps[1]):#vertup
        patches.append(x_i[0:ps[0],p-ps[1]:p,:])
        patches.append(x_i[-ps[0]:,p-ps[1]:p,:])
    return patches
    
def patch_to_edgeim(x_i, result, ps):
    result_img = np.zeros((x_i.shape[0],x_i.shape[1],6))
    loc = 0
    for p in range(0,x_i.shape[0]//2,ps[0]):#horizleft
        result_img[p:p+ps[0],0:ps[1],:] = result[loc]
        result_img[p:p+ps[0],-ps[1]:,:] = result[loc+1]
        loc += 2
        
    for p in range(x_i.shape[0],x_i.shape[0]//2,-ps[0]):#horizright
        result_img[p-ps[0]:p,0:ps[1],:] = result[loc]
        result_img[p-ps[0]:p,-ps[1]:,:] = result[loc+1]
        loc += 2
        
    for p in range(ps[1],x_i.shape[1]//2,ps[1]):#vertup
        result_img[0:ps[0],p:p+ps[1],:] = result[loc]
        result_img[-ps[0]:,p:p+ps[1],:] = result[loc+1]
        loc += 2
        
    for p in range(x_i.shape[1]-ps[1],x_i.shape[1]//2,-ps[1]):#vertup
        result_img[0:ps[0],p-ps[1]:p,:] = result[loc]
        result_img[-ps[0]:,p-ps[1]:p,:] = result[loc+1]         
        loc += 2
    return result_img
    
#----Load Data---

set_imgs = load.load_all(REAL_PATH,comp=comp)
print("Loading done")
print("Pre-processing")
x_img = set_imgs[1]
y_img = set_imgs[0]
x_img = proc._join(x_img,set_imgs[2:])  
_d = x_img[0].shape[2]

#hnv_imgs = load.load_data_simp([REAL_PATH+'hnv_ng'])
#x_img = proc._join(x_img,hnv_imgs[0:1],depth=2)


model = load_model(TRAIN_PATH+'/model.hdf5',compile=False)
print("Model loaded")

_w = patch_size[0]
_h = patch_size[1]
_w_out = _w

i=-1
for x_i in x_img:
    i+=1
    result_img = np.zeros((x_i.shape[0],x_i.shape[1],6))

    number = str(i).zfill(4)

    iters = x_i.shape[0]//_w
    iters = x_i.shape[1]//_h if iters < x_i.shape[1]//_h else iters
    
    for e in range((iters//2)-1):
        if e > 0:
            ww = (x_i.shape[0]-(_w))//2
            ww = ww if ww < _w else _w
            hh = (x_i.shape[1]-(_h))//2
            hh = hh if hh < _h else _h
            if ww == 0:
                x_i = x_i[:,hh:-hh,:]
            elif hh == 0:
                x_i = x_i[ww:-ww,:,:]
            else:
                x_i = x_i[ww:-ww,hh:-hh,:]
        else:
            x_i = x_i
        print(x_i.shape)
        resultedge = model.predict(np.array(edgeim_to_patch(x_i,patch_size)))
        resultedge = patch_to_edgeim(x_i,resultedge,patch_size)
        if e > 0:
            www = (result_img.shape[0]-resultedge.shape[0])//2
            hhh = (result_img.shape[1]-resultedge.shape[1])//2
            result_img[www:-www,hhh:-hhh,:] = resultedge
        else:
            result_img = resultedge
 

    result_img = classes.declassimg2(np.argmax(result_img-1,axis=2),map=True)
    misc.imsave(RESULT_PATH_BASE+"/"+number+".png",result_img)
    

