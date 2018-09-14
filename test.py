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
if arg.no_depth:
    x = proc.cat_imgs(rgb,rgb)### NO DEPTH
else:
    x = proc.cat_imgs(rgb,d)### WITH DEPTH
del rgb, d

#----KERAS ENV-------
os.environ["THEANO_FLAGS"]='device=cuda0'
#sys.setrecursionlimit(50000)

from keras.models import Model, load_model

print("Loading Model")
model = load_model(arg.output_folder+'/model.hdf5', compile=False)
train_args = load.get_args(arg.output_folder)
arg.mean = np.array(train_args['mean'])
p_s = arg.patch_size
#_!_!_!_!_!_!_RUN NETWORK_!_!_!_!_!_
print("Running Network")

for num,i in enumerate(x):
    all_out = []
    for o_s in range(0,p_s//2,p_s//(4*2)):
        padded_i = i
        if o_s != 0:
            padded_i = np.pad(i,((o_s,0),(o_s,0), (0,0)),'constant',constant_values=0)
        print("Generating: "+str(num))
        input,steps = proc.patch_factor_resize(padded_i,p_s)
        result = model.predict_generator(proc.generate_predict_patch(input, p_s, arg.batch_size, arg.mean), steps=steps//arg.batch_size+1)
        output = np.zeros((input.shape[:2])+(result.shape[3],))
        for x in range(0,input.shape[0],p_s):
            for y in range(0,input.shape[1],p_s):
                ind = (x*input.shape[1]//p_s + y)//p_s
                output[x:x+p_s,y:y+p_s,:] = result[ind]
        if o_s != 0:
            output = output[o_s:,o_s:,:]
        output = output[:i.shape[0],:i.shape[1],:]
        all_out.append(output)
    all_out = np.mean(np.array(all_out),axis=0)
    all_out = proc.uncategorize_img(all_out, [4,2,6,1,3,7], 7)
    misc.imsave(arg.output_folder+"/test"+str(num)+".png",all_out)