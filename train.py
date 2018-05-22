import os
import sys
import numpy as np
from scipy import misc
import handlers.args as args
import handlers.load as load
import handlers.proc.Process as proc

arg = args.get_args()

#----Train Result Paths---
OUT_PATH = 'results/'
TRAIN_PATH = OUT_PATH+arg.name
if os.path.exists(TRAIN_PATH):
    if not arg.cont:
        print("Training path already exists")
        sys.exit(0)
else:
    if arg.cont:
        print("Model does not exist")
        sys.exit(0)
    os.makedirs(TRAIN_PATH)

#----Load Data---
rgb = load.load_data("data/rgb")
d = load.load_data("data/d")

x = proc.cat_imgs(rgb,d)
y = load.load_data("data/y")

xy = proc.setup(x,y,arg.patch_size,arg.batch_size)

#----KERAS ENV-------
os.environ["THEANO_FLAGS"]='device='+arg.device
sys.setrecursionlimit(50000)

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model

#----MODEL MODULE---
from models import loader
module = loader.load(arg.model)

if cont:
    print("Loading Model")
    model = load_model(TRAIN_PATH+'/model.hdf5')
else:
    print("Generating Model")
    model = module.build((patch_size+(x_img[0].shape[2],)),nclasses=len(label))
    print("Compiling Model")
    model.compile('RMSprop', 'categorical_crossentropy')
    


checkpointer = ModelCheckpoint(filepath=TRAIN_PATH+'/model.hdf5', verbose=0, save_best_only=False)
breakPateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
plotter = plotting.PlotLoss()

print("Setting up Network")

model.fit_generator(proc.generate_patch(),steps_per_epoch=64,epochs=500000,callbacks=[checkpointer,plotter,breakPateau],use_multiprocessing=True)
