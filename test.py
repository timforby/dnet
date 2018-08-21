import numpy as np
import os
import handlers.args as args
import handlers.load as load
from handlers.proc import Process as proc
from scipy import misc

arg = args.get_args()

#----Load Data---
print("Loading Images: "+arg.input_folder)
rgb = load.load_data(arg.input_folder+"/rgb_ng")
d = load.load_data(arg.input_folder+"/d_ng")
print("Loaded "+ str(len(rgb)) + " images")
x = proc.cat_imgs(rgb,d)
del rgb, d

#----KERAS ENV-------
os.environ["THEANO_FLAGS"]='device=cuda0'
#sys.setrecursionlimit(50000)

from keras.models import Model, load_model

print("Loading Model")
model = load_model(arg.output_folder+'/model.hdf5', compile=False)

#_!_!_!_!_!_!_RUN NETWORK_!_!_!_!_!_
print("Running Network")

for num,i in enumerate(x): 
    print("Generating: "+str(num))
    row = (1+i.shape[0]//arg.patch_size[0])*arg.patch_size[0]
    col = (1+i.shape[1]//arg.patch_size[1])*arg.patch_size[1]
    input = np.zeros((row,col)+(i.shape[2],))
    input[:i.shape[0],:i.shape[1],:] = i
    steps = input.shape[0]*input.shape[1]//(arg.patch_size[0]**2)//arg.batch_size
    result = model.predict_generator(proc.generate_predict_patch(input, arg.patch_size, arg.batch_size), steps=steps+1)
    output = np.zeros((input.shape[:2])+(result.shape[3],))
    for x in range(0,row,arg.patch_size[0]):
        for y in range(0,col,arg.patch_size[0]):
            ind = (x*col//arg.patch_size[0] + y)//arg.patch_size[0]
            output[x:x+arg.patch_size[0],y:y+arg.patch_size[1],:] = result[ind]
        
    #output = np.reshape(result, (row, col, -1))
    output = proc.uncategorize_img(output, [4,2,6,1,3,7], 7)
    misc.imsave(arg.output_folder+"/test"+str(num)+".png",output)