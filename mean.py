import numpy as np
import handlers.args as args
import handlers.load as load
from handlers.proc import Process as proc

arg = args.get_args()

#----Load Data---
print("Loading Images")
rgb = load.get_filenames(arg.input_folder)

siz = [0,0,0]
sum = [0,0,0]
for x in rgb:
    x = load.load_img(arg.input_folder, x)
    for i in range(3):
        sum[i] += np.sum(x[:,:,i])
        siz[i] += x.shape[0]*x.shape[1]
        
mean = np.array(sum)/np.array(siz)
f = open(arg.input_folder+"/mean.txt","w")
f.write(str(mean))
f.close()
