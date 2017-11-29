import os
import sys
import numpy as np
import argparse
from scipy import misc

ar = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ar.add_argument('gpu', type = int, help = "Please enter GPU to use")
ar.add_argument('name', type = str, help = "Please training name - for save folder")
ar.add_argument('model_name', type = str, help = "Please enter model type - unet,mynet,resnet50")
ar.add_argument('patch_size', type = int, help = "Please enter model patch size")
ar.add_argument('-o','--patch_size_out', type = int, help = "Please enter model output patch size")
ar.add_argument('-b','--batch_size', type = int, help = "Please enter model batch size")
ar.add_argument('--predef', action='store_true', help='Option defining whether to use predefined patches')
ar.add_argument('--load_balance', action='store_true', help='Option defining whether have load balancing')
ar.add_argument('-l','--labels', action='append', help="Option to select specific training labels \nOTHER = 0\nTREE = 1\nBUILDING = 2\nCAR = 3\nVEG = 4\nGROUND = 5")
ar.add_argument('--continue_model', action='store_true', help='Option defining whether to continue existing model')
ar.add_argument('--dilate', action='store_true', help='Transfer weights into dilation convolutions')
args = ar.parse_args()
device = 'cuda'+str(args.gpu)
name = args.name
model_name = args.model_name
patch_size = (args.patch_size,args.patch_size)
patch_size_out = args.patch_size_out
if patch_size_out:
    patch_size_out = (patch_size_out,patch_size_out)
predef = args.predef
cont = args.continue_model
single_pixel_out = False
label = args.labels
if label ==  None:
    label = [0,1,2,3,4,5]
else:
    label = list(map(int, label))
if len(label)<6:
    label = [0]+label
print("Using labels: "+','.join(map(str,label)))
if model_name=='resnet50':
    single_pixel_out = True
batch_size = args.batch_size
if batch_size == None:
    batch_size = 32
    
dilate = args.dilate
#----Image Paths---

BASE_PATH = '../../data/'
REAL_PATH = BASE_PATH+'real/'
OUT_PATH = BASE_PATH+'out/'
     
TRAIN_PATH = OUT_PATH+name
if not cont:
    if os.path.exists(TRAIN_PATH):
        i = 0
        while os.path.exists(TRAIN_PATH):
            TRAIN_PATH = OUT_PATH+name+'_'+str(i)
            i += 1
    os.makedirs(TRAIN_PATH)
else:
    if not os.path.exists(TRAIN_PATH):
        print("Model does not exist")
        sys.exit(0)



#----KERAS ENV-------

os.environ["THEANO_FLAGS"]='device='+device
sys.setrecursionlimit(50000)

import load,proc,classes,math,plotting,model
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model


#----Load Data---
if not predef:
    set_imgs= load.load_all(REAL_PATH)
    set_imgs2= load.load_all(REAL_PATH,comp=True)
    print("Loading Done")
    print("Preprocessing")
    x_img = set_imgs[1]
    y_img = proc.preprocess_ys(set_imgs[0],x_img,labels=label)
    x_img = proc._join(x_img,set_imgs[2:])  
    x_img_val = set_imgs2[1]
    y_img_val = proc.preprocess_ys(set_imgs2[0],x_img_val,labels=label)
    x_img_val = proc._join(x_img_val,set_imgs2[2:]) 
    
    #hnv_imgs = load.load_data_simp([REAL_PATH+'hnv',REAL_PATH+'hnv_ng'])
    #x_img = proc._join(x_img,hnv_imgs[0:1],depth=2)
    #x_img_val = proc._join(x_img_val,hnv_imgs[1:2],depth=2)


if not cont or dilate:
    print("Generating Model")
    model = model.gen((patch_size+(x_img[0].shape[2],)),model_name,nclasses=len(label),p_out=patch_size_out)
    print("Compiling Model")
    #plot_model(model, to_file=TRAIN_PATH+'/model.png',show_shapes=True,show_layer_names=True)
    model.compile('RMSprop', 'categorical_crossentropy')
else:
    print("Loading Model")
    model = load_model(TRAIN_PATH+'/model.hdf5')


checkpointer = ModelCheckpoint(filepath=TRAIN_PATH+'/model.hdf5', verbose=0, save_best_only=False)
breakPateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
plotter = plotting.PlotLoss(TRAIN_PATH, x_img, y_img, x_img_val,y_img_val,patch_size,patch_out=patch_size_out,labels=label) if not predef and not single_pixel_out else plotting.PlotLoss(TRAIN_PATH, None, None, None, None, None, plot_images=False)

print("Setting up Network")

generator = proc.gen_aug(x_img,y_img,patch_size,patch_size_out,batch_size=batch_size,single_pixel=single_pixel_out,labels=label) if not predef else proc.gen_predef(REAL_PATH,batch_size=batch_size)

model.fit_generator(generator,steps_per_epoch=64,epochs=500000,callbacks=[checkpointer,plotter,breakPateau],use_multiprocessing=True)
