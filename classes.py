import numpy as np
from enum import Enum

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    LIGHTBLUE = 4
    WHITE = 5
    
'''class Category(Enum):
    OTHER = 0
    TREE = 1
    BUILDING = 2
    CAR = 3
    VEG = 4
    GROUND = 5'''

    
class Category(Enum):
    OTHER = 1
    TREE = 2
    BUILDING = 4
    CAR = 3
    VEG = 6
    GROUND = 7
    
def to_class(val):
    if np.array_equal(val, [1,0,0]):
        return Category.OTHER
    elif np.array_equal(val, [0,1,0]):
        return Category.TREE
    elif np.array_equal(val, [0,0,1]):
        return Category.BUILDING
    elif np.array_equal(val, [1,1,0]):
        return Category.CAR
    elif np.array_equal(val, [0,1,1]):
        return Category.VEG
    elif np.array_equal(val, [1,1,1]):
        return Category.GROUND
     
def is_cat(y,CAT):
    val = y[0,0]
    return CAT == to_class(val)
    
def get_cat(y):
    val = y[0,0]
    return to_class(val)
     
def to_color(val, binary=False):
    res = np.argmax(val)
    if binary:
        res = binary.value if res == 1 else -1
    if res == -1:
        return [0,0,0]
    elif res == 0:
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

    
def to_bin(val,sel):
    val = int(val[0])
    sel = sel.value
    return 1 if val == sel else 0
    
def multi(ys,cats):
    y_comp = np.copy(ys)
    for k, v in getmap(cats):              
        y_comp[ys==k]= v
    return y_comp  

def declassimg(ys,binary=None):
    y_out = np.empty((ys.shape[0],ys.shape[1],3))
    for x in range(ys.shape[0]):
        for y in range(ys.shape[1]):
            y_out[x,y] = to_color(ys[x,y],binary)
    return y_out

def declassimg_multi(ys, cats=None):
    ys = np.argmax(ys,axis=2)
    y_comp = np.copy(ys)
    
    for i in range(len(cats)):
        y_comp[ys==i+1]= cats[i].value

    y_out = np.empty((ys.shape[0],ys.shape[1],3))
    y_out[:,:,0] = np.mod(y_comp,2)
    y_out[:,:,1] = np.mod((y_comp//2),2)
    y_out[:,:,2] = np.mod((y_comp//4),2)
    return y_out

def declassimg2(ys, map=False,cats=None):
    yss = np.copy(ys)
    if cats:
        i = 0
        for cat in cats:
            yss[ys==i]=cat
            i+=1
        ys = np.copy(yss)
    ys = np.around(ys)
    ys = ys.astype(int)
    if map:
        y_comp = np.copy(ys)
        for k, v in getmap():              
            y_comp[ys==v]= k
        ys = np.copy(y_comp)
    if len(ys.shape) ==3:
        ys = np.reshape(ys,(ys.shape[:2]))
    y_out = np.empty((ys.shape[0],ys.shape[1],3))
    y_out[:,:,0] = np.mod(ys,2)
    y_out[:,:,1] = np.mod((ys//2),2)
    y_out[:,:,2] = np.mod((ys//4),2)
    return y_out

def classimg(ys,map=False,labels=[0,1,2,3,4,5]):
    y_out = ys[:,:,0]+(2*ys[:,:,1])+(4*ys[:,:,2])
    y_comp = np.copy(y_out)

    if map:
        for k, v in getmap(labels=labels):              
            y_comp[y_out==k]= v
    y_comp = np.expand_dims(y_comp,axis=2)
    return y_comp
   
def getmap(labels=None):
    keys=[1,2,4,3,6,7]
    vals=[0,0,0,0,0,0]
    if labels:
        for l in labels:
            vals[l]=l
    else:
        vals=[0,1,2,3,4,5]
    return zip(keys,vals)
