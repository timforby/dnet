import numpy as np
import os
import handlers.args as args
import handlers.load as load
from handlers.proc import Process as proc

arg = args.get_args()

#----Load Data---
print("Loading Images")
rgb = load.load_data(arg.input_folder+"/rgb")
rgb_mean = load.get_mean(arg.input_folder+"/rgb")

if arg.no_depth:### NO DEPTH
    x = proc.cat_imgs(rgb,rgb)
    arg.mean = np.concatenate([rgb_mean,rgb_mean])[:x[0].shape[2]]
else:### --- WITH DEPTH
    d_mean = load.get_mean(arg.input_folder+"/d")
    d = load.load_data(arg.input_folder+"/d")
    x = proc.cat_imgs(rgb,d)
    del d
    arg.mean = np.concatenate([rgb_mean,d_mean])[:x[0].shape[2]]
del rgb
y = load.load_data(arg.input_folder+"/y")

#validation
rgb_ng = load.load_data(arg.input_folder+"/rgb_ng",end=1)
if arg.no_depth:### NO DEPTH
    x_ng = proc.cat_imgs(rgb_ng,rgb_ng)
else:### --- WITH DEPTH
    d_ng = load.load_data(arg.input_folder+"/d_ng")
    x_ng = proc.cat_imgs(rgb_ng,d_ng)
    del d_ng
del rgb_ng
y_ng = load.load_data(arg.input_folder+"/y_ng")

proc.setup(x,y, x_ng, y_ng, arg.patch_size,arg.batch_size, arg.mean)
#proc.setup(x,y, None, None, arg.patch_size,arg.batch_size)

del y, y_ng

#----KERAS ENV-------
os.environ["THEANO_FLAGS"]='device=cuda'+str(arg.gpu)
#sys.setrecursionlimit(50000)

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model


#----COMPILE MODEL---
if arg.cont:
    print("Loading Model")
    model = load_model(arg.output_folder+'/model.hdf5')
else:
    #----MODEL MODULE---
    from models import loader
    module = loader.load(arg.model)
    print("Generating Model")
    model = module.build(proc.input_shape,len(proc.y_classes))
    print("Compiling Model")
    model.compile('RMSprop', 'categorical_crossentropy')
    

#----CHECKPOINT CALLBACKS -----
checkpointer = ModelCheckpoint(filepath=arg.output_folder+'/model.hdf5', verbose=0, save_best_only=False)
#PLOT
from handlers.plots import PlotLoss as plot
plotter = plot(arg, proc)

#_!_!_!_!_!_!_RUN NETWORK_!_!_!_!_!_
print("Running Network")
model.fit_generator(proc.generate_patch(augment=arg.augment),steps_per_epoch=64,epochs=500000,callbacks=[checkpointer,plotter],use_multiprocessing=False)
