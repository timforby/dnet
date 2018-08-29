import math
import random
from scipy.ndimage import rotate
import cv2


def augment_patch(x,y,patch_size):
    a = np.concatenate((x,y) axis=2)
    #Rotation
    angle = random.randint(0,30)*3
    if angle >0:
        a = rotate(a,angle)
        row_start = a.shape[1]//2-patch_size//2
        row_end = a.shape[1]//2-patch_size//2
        col_start = a.shape[0]//2-patch_size//2
        col_end = a.shape[0]//2-patch_size//2
        a = a[col_start:col_end,row_start:row_end,:]
    
    #Resize
    zoom = random.randint(75,100)/100
    a = np.resize(a, (a.shape[0]*zoom,a.shape[1]*zoom, a.shape[2]))
    a = cv2.resize(a,(patch_size,patch_size,a.shape[2]), interpolation = cv2.INTER_LINEAR)
    
    #Flipvert
    if random.randint(0,1) == 1:
        a = flipud(a)
    if random.randint(0,1) == 1:
        a = fliplr(a)
        
    return a[:,:,:x.shape[2]],a[:,:,x.shape[2]:]