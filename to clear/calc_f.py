import load
import numpy as np
import argparse
from scipy import misc
from sklearn.metrics import f1_score
from enum import Enum




class Category(Enum):
    TREE = 2
    BUILDING = 4
    CAR = 3
    VEG = 6
    GROUND = 7

ar = argparse.ArgumentParser()
ar.add_argument('path', type = str, help = "Please enter path to folder with images")
args = ar.parse_args()
MODEL_NAME = args.path
GROUND = False
BASE_PATH = '../../data/'
REAL_PATH = BASE_PATH+'real/'
#image_paths = ['../../data/small/real/test','../../data/small/out/test']
set_imgs = load.load_data(REAL_PATH, OTHER=MODEL_NAME,GROUND=GROUND)

f = open(MODEL_NAME+'/f1score.txt','w')
def p(s):
    print(s)
    f.write(s+'\n')


gs = set_imgs[0]
rs = set_imgs[3]
TTp = 0
TFp = 0
TTn = 0
TFn = 0
for i in range(len(gs)):
    if not GROUND:
        gres = misc.imresize(gs[i],rs[i].shape,interp='nearest')//255
    else:
        gres = gs[i]
    g = gres[:,:,0]+(2*gres[:,:,1])+(4*gres[:,:,2])
    r = rs[i][:,:,0]+(2*rs[i][:,:,1])+(4*rs[i][:,:,2])
    #p(f1_score(g.flatten(),r.flatten(),[7,4,6,2,3,1],average='micro'))
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    labels = [2,4,3,6,7]
    p("Image: "+str(i))
    for L in labels:
        #init labelwise scoring vals
        lTp = 0
        lFp = 0
        lTn = 0
        lFn = 0
        # create result image where 1=label, 0=non label
        rb = (r==L).astype(int)
        # create ground image where 2=label, 0=non label
        gb = 2*((g==L).astype(int))
        # result: -1 = truepos, 1 = falsepos, -2= falseneg, 0 = trueneg
        res = rb-gb
        res = res.flatten()
        # result: 0 = falseneg, 1=truepos, 2=trueneg, 3=falsepos
        res +=2
        res = np.bincount(np.append(res,[0,1,2,3]))   
        
        lTp += (res[1]-1)//1
        lFn += (res[0]-1)//1
        lTn += (res[2]-1)//1
        lFp += (res[3]-1)//1
        try:
            #p(str(lTp)+" : "+str(lFn)+" : "+str(lTn)+" : "+str(lFp))
            lPrecision = lTp/(lTp+lFp)
            lRecall = lTp/(lTp+lFn)
            lF1 = 2*(lPrecision*lRecall)/(lPrecision+lRecall)
        except:
            lF1 = 0
        p(Category(L).name+": "+str(lF1))
        Tp += lTp
        Tn += lTn
        Fp += lFp
        Fn += lFn
    Precision = Tp/(Tp+Fp)
    Recall = Tp/(Tp+Fn)
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    p("Total F1: "+str(F1))
    p("==================")
    TTp += Tp
    TTn += Tn
    TFp += Fp
    TFn += Fn
    #'''
p("==================")
p("==================")
TPrecision = TTp/(TTp+TFp)
TRecall = TTp/(TTp+TFn)
TF1 = 2*(TPrecision*TRecall)/(TPrecision+TRecall)
p("Overall F1: "+str(TF1))
p("==================")
p("==================")
f.close()
