import os
import sys
import numpy as np
import argparse
from scipy import misc
import load,classes

ar = argparse.ArgumentParser()
ar.add_argument('name', type = str, help = "Please enter folder name")
args = ar.parse_args()
name = args.name

PATH =  '../../data/out/'+name+'/'

ranges = range(-50,100,50)
offsets = []
for xoff in ranges:
    for yoff in ranges:
        if yoff == xoff:
            offsets.append((xoff,yoff))
paths = []


for off in offsets:
    paths.append(PATH+str(off[0])+'_'+str(off[1]))


imgs_base = load.load_data_simp([PATH+'0_0'])

RESULT = PATH+'result/'
if not os.path.exists(RESULT):
    os.makedirs(RESULT)


for i in range(len(imgs_base[0])):

    imgs = load.load_data_simp(paths,start=i,end=i+1)

    images = []

    #pad and crop

    x = 0
    for offset in offsets:
        x_i = imgs[x][0]
        for axis in range(2):
            beffoffset = 0 if offset[axis] < 0 else offset[axis]
            aftoffset = 0 if offset[axis] > 0 else offset[axis]*-1
            
            if axis ==0:
                x_i = np.pad(x_i,((beffoffset,aftoffset),(0,0),(0,0)),'constant')
                x_i = x_i[:-beffoffset,:,:] if beffoffset > 0 else x_i[aftoffset:,:,:]
            else:
                x_i = np.pad(x_i,((0,0),(beffoffset,aftoffset),(0,0)),'constant')
                x_i = x_i[:,:-beffoffset,:] if beffoffset > 0 else x_i[:,aftoffset:,:]
        x_i = classes.classimg(x_i)
        images.append(x_i)
        x +=1
    
    '''
    result = np.copy(images[0])   
    for col in range(images[0].shape[0]):
        for row in range(images[0].shape[1]):
            pixels = np.zeros((len(offsets)),dtype=int)
            for j in range(len(offsets)):
                print(images[j][col,row])
                pixels[j] = images[j][col,row]
            pix = np.bincount(pixels).argmax()
            pix = pix if pix > 0 else pixels[orig_offset]
            result[col,row] = pix
    '''
    concat = np.concatenate(images,axis=2).astype(int)
    result = np.apply_along_axis(lambda x: np.bincount(x,minlength=6)[1:].argmax()+1, axis=2, arr=concat)


    result_img = classes.declassimg2(result)
    number = str(i).zfill(4)
    misc.imsave(RESULT+number+".png",result_img)
    print('done with: '+str(i))



'''    i=-1
    for x_i in x_img:
        i+=1
        result_img = np.zeros(x_i.shape)
        for offset in offsets:
            RESULT_PATH = RESULT_PATH_BASE+'/'+str(offset[0])+'_'+str(offset[1])+'/'
            if not os.path.exists(RESULT_PATH):
                os.makedirs(RESULT_PATH)

            number = str(i).zfill(4)#str(i*2+EVEN).zfill(4)
            if os.path.isfile(RESULT_PATH+number+".png"):
                print("Image already created: "+number)
                continue

            for axis in range(2):
                beffoffset = 0 if offset[axis] > 0 else offset[axis]*-1
                aftoffset = 0 if offset[axis] < 0 else offset[axis]
                if axis ==0:
                    x_i = np.pad(x_i,((beffoffset,aftoffset),(0,0),(0,0)),'constant')
                    x_i = x_i[:-beffoffset,:,:] if beffoffset > 0 else x_i[aftoffset:,:,:]
                else:
                    x_i = np.pad(x_i,((0,0),(beffoffset,aftoffset),(0,0)),'constant')
                    x_i = x_i[:,:-beffoffset,:] if beffoffset > 0 else x_i[:,aftoffset:,:]

            limit = 20
            padded_w = x_i.shape[0]+(_w-(x_i.shape[0]%_w))
            padded_h = x_i.shape[1]+(_h-(x_i.shape[1]%_h))
            max = (padded_w)*(padded_h)/(_w*_h)
            result = model.predict_generator(proc.generator_predict_patch(x_i,_w,limit),math.ceil(max/limit),max_queue_size=4,use_multiprocessing=True, verbose=1)
            result = proc.reshape_patches(result,(padded_w-(_w-_w_out),padded_h-(_w-_w_out),6),_w_out)
            #result = proc.reshape_patches(result,(padded_w,padded_h,6),_w)
            result_img += result
        result = result_img/len(offsets)
        result = classes.declassimg2(np.argmax(result,axis=2),map=True)
        misc.imsave(RESULT_PATH_BASE+'/'+number+".png",result[:x_i.shape[0],:x_i.shape[1]])
'''
