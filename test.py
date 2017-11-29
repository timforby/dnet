import os
import sys
import numpy as np
import argparse
from scipy import misc
from densecrf import crf

ar = argparse.ArgumentParser()
ar.add_argument('gpu', type = int, help = "Please enter GPU to use")
ar.add_argument('name', type = str, help = "Please model name")
ar.add_argument('patch_size', type = int, help = "Please enter model patch size")
ar.add_argument('interval', type = int, help = "Please enter processing interval")
ar.add_argument('--comp', action='store_true', help='Option defining whether process competition images')
ar.add_argument('-l','--labels', action='append', help="Option to select specific training labels \nOTHER = 0\nTREE = 1\nBUILDING = 2\nCAR = 3\nVEG = 4\nGROUND = 5")
args = ar.parse_args()
device = 'cuda'+str(args.gpu)
name = args.name
interval = args.interval
patch_size = (args.patch_size,args.patch_size)
comp = args.comp
labels = args.labels
if labels:
    labels = list(map(int, labels))
    labels = [0]+labels
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
'''
if os.path.exists(RESULT_PATH_BASE):
    i = 0
    while os.path.exists(RESULT_PATH_BASE):
        RESULT_PATH_BASE = TRAIN_PATH+append+'_'+str(i)
        i += 1
'''
if not os.path.exists(RESULT_PATH_BASE):
    os.makedirs(RESULT_PATH_BASE)
RESULT_PATH_BASE = RESULT_PATH_BASE + '/'+ str(interval)

os.makedirs(RESULT_PATH_BASE)
os.makedirs(RESULT_PATH_BASE+'/crf')
os.makedirs(RESULT_PATH_BASE+'/avg')

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
    
def patch_to_edgeim(x_i, result, ps,out_depth):
    result_img = np.zeros((x_i.shape[0],x_i.shape[1],out_depth))
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
out_depth = None
ranges = range(-(patch_size[0]//2),(patch_size[0]//2),interval)
#ranges = range(0,(patch_size[0]//2),interval)
offsets = []
for xoff in ranges:
    for yoff in ranges:
        if yoff == xoff:
            offsets.append((xoff,yoff))


def test_img(x_i,i):
    
    concat = []
    number = str(i).zfill(4)#str(i*2+EVEN).zfill(4)
    
    result_img = np.zeros((x_i.shape[0],x_i.shape[1],6))
    result_cen = np.zeros((x_i.shape[0],x_i.shape[1],6))
    result_edg = np.zeros((x_i.shape[0],x_i.shape[1],6))
    img_count = 0
    edg_count = 0
    for rot in range(1):
        x_i_r = np.rot90(x_i,rot)
        for offset in offsets:        
            prepad = np.copy(x_i_r)
            for axis in range(2):
                beffoffset = 0 if offset[axis] > 0 else offset[axis]*-1
                aftoffset = 0 if offset[axis] < 0 else offset[axis]
                if axis ==0:
                    prepad = np.pad(prepad,((beffoffset,aftoffset),(0,0),(0,0)),'constant')
                    prepad = prepad[:-beffoffset,:,:] if beffoffset > 0 else prepad[aftoffset:,:,:]
                else:
                    prepad = np.pad(prepad,((0,0),(beffoffset,aftoffset),(0,0)),'constant')
                    prepad = prepad[:,:-beffoffset,:] if beffoffset > 0 else prepad[:,aftoffset:,:]

            limit = 20
            padded_w = x_i_r.shape[0]+(_w-(x_i_r.shape[0]%_w))
            padded_h = x_i_r.shape[1]+(_h-(x_i_r.shape[1]%_h))
            max = (padded_w)*(padded_h)/(_w*_h)
            result = model.predict_generator(proc.generator_predict_patch(prepad,_w,limit),math.ceil(max/limit),max_queue_size=4,use_multiprocessing=True, verbose=1)
            result = proc.reshape_patches(result,(padded_w-(_w-_w_out),padded_h-(_w-_w_out),result.shape[3]),_w_out)
            out_depth = result.shape[2]
            if offset==offsets[0] and result_img.shape[2] != out_depth:
                result_img = np.delete(result_img,np.s_[out_depth:],2)
                result_cen = np.delete(result_cen,np.s_[out_depth:],2)
            
            for axis in range(2):
                beffoffset = 0 if offset[axis] < 0 else offset[axis]
                aftoffset = 0 if offset[axis] > 0 else offset[axis]*-1
                if axis ==0:
                    result = np.pad(result,((beffoffset,aftoffset),(0,0),(0,0)),'constant',constant_values=-1)
                    result = result[:-beffoffset,:,:] if beffoffset > 0 else result[aftoffset:,:,:]
                else:
                    result = np.pad(result,((0,0),(beffoffset,aftoffset),(0,0)),'constant',constant_values=-1)
                    result = result[:,:-beffoffset,:] if beffoffset > 0 else result[:,aftoffset:,:]
            result = result[:x_i_r.shape[0],:x_i_r.shape[1]]
            
            result = np.rot90(result,rot*-1)
            img_count += 1
            result_img += result  
            
        
            #concat.append(np.expand_dims(np.apply_along_axis(sp_argmax,axis=2,arr=result),axis=2))
        
        #add edge effect
    resultedge = model.predict(np.array(edgeim_to_patch(x_i_r,patch_size)))
    result_edg = patch_to_edgeim(x_i_r,resultedge,patch_size,out_depth)
        #result_edg += np.rot90(resultedge,rot*-1)
        #edg_count += 1
    
    #result_edg /= edg_count #avg edge
    #for x in range(len(offsets)//3):
        #concat.append(np.expand_dims(np.apply_along_axis(sp_argmax,axis=2,arr=result_edg),axis=2))

    
    #result_2 = np.concatenate(concat,axis=2).astype(int)
    #result_2 = np.apply_along_axis(lambda x: np.bincount(x,minlength=7)[1:].argmax(), axis=2, arr=result_2)
    #result_2 = classes.declassimg2(result_2, map=True)
    #misc.imsave(RESULT_PATH_BASE+'/concat/'+number+".png",result_2)

    mul = (img_count//3)
    if mul == 0:
        mul = 1
    avgs = result_img/img_count
    result_edg[_w:-_w,_h:-_h,:] = avgs[_w:-_w,_h:-_h,:]
    edge = result_edg
    
    edge *= mul #weighted avgs with edges
    
    avgs = edge + avgs #weights edges plus center avgs
    avgs /= (mul+1) #all averages
    result_1 = avgs
    #result_1 = result_edg
    #result_edg[_w:-_w,_h:-_h,:] = result_1[_w:-_w,_h:-_h,:]
    #result_1 = result_edg
    #resultcent = np.copy(result_1)
    #resultcent[_w:-_w,_h:-_h,:] = result_cen[_w:-_w,_h:-_h,:]
    #result_edg[_w:-_w,_h:-_h,:] = result_1[_w:-_w,_h:-_h,:]
    #result_1 += mul*result_edg
    #result_1 += mul*resultcent
    #result_1 = result_1/((mul)+1)
    if labels:
       result_full = np.zeros((x_i.shape[0],x_i.shape[1],6))
       result_full[:,:,labels] = result_1
       result_1 = result_full  
    result_3 = crf(result_1,x_i)
    result_3 = classes.declassimg2(result_3,map=True)
    result_2 = classes.declassimg2(np.argmax(result_1,axis=2),map=True)
    misc.imsave(RESULT_PATH_BASE+"/crf/"+number+".png",result_3)
    misc.imsave(RESULT_PATH_BASE+"/avg/"+number+".png",result_2)
    


i=-1
for x_i in x_img:
    i+=1
    test_img(x_i,i)
    

