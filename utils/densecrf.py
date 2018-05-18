import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax,unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import classes as cl
import cv2

def crf(labels,rgb,classes=None,ls=[0,1,2,3,4,5]):

    if classes==None:
        labels += np.amin(labels)
        labels /= np.amax(labels)
        classes = labels.shape[2]      
        c_img = np.rollaxis(labels,2)
        U = unary_from_softmax(c_img)
    else:
        c_img = cl.classimg(labels,map=True,labels=ls)
        c_img = np.reshape(c_img,(c_img.shape[0]*c_img.shape[1])).astype(int)
        U = unary_from_labels(np.array([0,3]),2,.7,zero_unsure=True)
        
    d = dcrf.DenseCRF2D(labels.shape[0], labels.shape[1], classes)

    d.setUnaryEnergy(U)
    
    PG = create_pairwise_gaussian((3,3),labels.shape[:2])
    d.addPairwiseEnergy(PG,3)
    
    PB = create_pairwise_bilateral((59,59), (13,13,13,13), rgb,chdim=2)
    d.addPairwiseEnergy(PB,6)
    
    Q = d.inference(5)
    map = np.argmax(Q, axis=0)
    map = np.reshape(map,labels.shape[:2])
    
    return map
