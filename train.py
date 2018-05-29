import numpy as np
import handlers.args as args
import handlers.load as load
from handlers.proc import Process as proc

arg = args.get_args()

#----Load Data---
print("Loading Images")
rgb = load.load_data(arg.input_folder+"rgb")
d = load.load_data(arg.input_folder+"d")

x = proc.cat_imgs(rgb,d)
y = load.load_data(arg.input_folder+"y")

rgb_ng = load.load_data(arg.input_folder+"rgb_ng")
d_ng = load.load_data(arg.input_folder+"d_ng")

x_ng = proc.cat_imgs(rgb_ng,d_ng)
y_ng = load.load_data(arg.input_folder+"y_ng")

proc.setup(x,y, x_ng, y_ng, arg.patch_size,arg.batch_size)
#proc.setup(x,y, None, None, arg.patch_size,arg.batch_size)

del x,y,x_ng,y_ng,d,d_ng

#----KERAS ENV-------
#os.environ["THEANO_FLAGS"]='device=cuda'+str(arg.gpu)
#sys.setrecursionlimit(50000)

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import plot_model

#----MODEL MODULE---
from models import loader
module = loader.load(arg.model)

#----COMPILE MODEL---
if arg.cont:
    print("Loading Model")
    model = load_model(arg.output_folder+'/model.hdf5')
else:
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
print("Setting up Network")
model.fit_generator(proc.generate_patch(),steps_per_epoch=64,epochs=500000,callbacks=[checkpointer,plotter],use_multiprocessing=False)
