import numpy as np
import os
from sklearn.preprocessing import label_binarize as lb

def join_imgs(xs,ys,depth=1):
	return [np.concatenate((x,y[:,:,:depth],axis=3)) for x,y in zip(xs,ys)]

def categorize_img(y, labels):
	y = to_single_channel(y,labels)
	_y = y.reshape((y.size,1))
	_mx = _y.max()
	_y = lb(_y,range(_sz))
	_y = _y.reshape((y.shape[:2]+(_sz,))).astype(int)
	_y = _y[:,:,labels]
	return _y

def categorize_imgs(ys,labels):
	return [categorize_img(y,labels) for y in ys] 

def to_single_channel(y, labels):
    y_res = y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])
    y_comp = np.copy(y_res)

    if map:
        for k, v in getmap(labels=labels):              
            y_comp[y_res==k]= v
    y_comp = np.expand_dims(y_comp,axis=2)
    return y_comp
	
