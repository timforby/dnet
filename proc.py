import numpy as np
import os
from multiprocessing import Process, Queue
from scipy import misc
from scipy import ndimage
from sklearn.preprocessing import label_binarize
import classes,math
import time

#FROM KERAS
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def reshape_patches(patches, shape, w, row_first=True):
    patches = np.concatenate(patches)
    major = 0 if row_first else 1
    img = np.zeros((shape))
    #img = np.zeros((shape[1],shape[0],shape[2]))
    #print(patches.size)
    width = shape[1]
    for i in range(0,patches.shape[0],w):
        sec = i
        #print("--- patch: "+str((i,i+w))+" from "+str(patches.shape))
        patch = patches[i:i+w,0:w]
        img_1 = i%width
        img_0 = (i//width)*w
        #print("+++ patch: "+str(((img_0,img_0+w),(img_1,img_1+w)))+" into "+str(img.shape))
        img[img_0:img_0+w,img_1:img_1+w]=patch

    return img

def _join(x,xs,depth=1):
    for x_ in xs:
        x = merge(x,x_,depth)#TO CHANGE
    return x
def merge(xs,xs_,depth):
    xm = []
    for x,x_ in zip(xs,xs_):

        xm.append(np.concatenate((x,x_[:,:,:depth]),axis=2))
    return xm
def catimg(ys,n_classes):
    y_out = np.empty((ys.shape[0],ys.shape[1],n_classes))
    for x in range(ys.shape[0]):
        for y in range(ys.shape[1]):
            y_out[x,y] = to_categorical(ys[x,y],n_classes)
            
    return y_out        
        
def gen_kernel(img,w,h,x,y,center=True):
    if center:
        xoff = w//2
        yoff = h//2    
        l = x - xoff
        r = x + xoff + 1   
        u = y - yoff
        d = y + yoff + 1
    else:
        l = x
        r = x + w
        u = y
        d = y + h  
    return img[l:r,u:d,:]
    
def get_xy(value, rows):
    return int(value%rows),int(value/rows)

def generator_predict(x_i,w,h,max,limit):
    while True:
        patches = []
        mid = (w//2)
        pad = ((mid,mid),(mid,mid),(0,0))
        x_i = np.pad(x_i,pad,'constant')
        for x_pixel in range(x_i.shape[0]-w+1):  
            for y_pixel in range(x_i.shape[1]-h+1):              
                patches.append(gen_kernel(x_i,w,h,x_pixel,y_pixel,center=False))
                if (x_pixel+1)*(y_pixel+1)==max or len(patches)>=limit:
                    yield (np.array(patches))
                    patches = []

def generator_predict_patch(x_i,p_size,limit,new_p_size=0):
    if new_p_size == 0:
        new_p_size=p_size
    patches = []
    p_w = p_size-(x_i.shape[0] % p_size)
    p_h = p_size-(x_i.shape[1] % p_size)
    pad = ((0,p_w),(0,p_h),(0,0))
    x_i = np.pad(x_i,pad,'constant')
    #print(x_i.shape)
    while True:
        i = 0
        for x_pixel in range(0,x_i.shape[0]-p_size+1,new_p_size):  
            for y_pixel in range(0,x_i.shape[1]-p_size+1,new_p_size):
                y_patch = gen_kernel(x_i,p_size,p_size,x_pixel,y_pixel,center=False)
                #print(str(y_patch.shape)+"  w:"+str(w)+"  h:"+str(h))
                patches.append(y_patch)
                i += 1
                #print(i)
                if (x_pixel*y_pixel)>=((x_i.shape[0]-p_size)*(x_i.shape[1]-p_size)) or len(patches)==limit:
                    yield (np.array(patches))
                    patches = []
                    

def generator_predict_para(x_i,w,h,max,limit):
    CPU = 2
    section = limit//CPU + 1

    while True:
        patches = []
        mid = (w//2)
        pad = ((mid,mid),(mid,mid),(0,0))
        x_i = np.pad(x_i,pad,'constant')
        q = Queue()
        procs = []
        end =(x_i.shape[0]-w+1)*(x_i.shape[1]-h+1)
        for pixel in range(0,end,section):
            p = Process(target=gen_patch_cpu, args=(x_i,w,h,pixel,section,q))
            p.start()
            procs.append(p)
            if (len(procs) < CPU) and (pixel+section < end):
                continue
            
            for i in procs:
                patches += q.get()
            
            for i in procs:
                i.join()
            
            procs = []
            yield(np.array(patches))
            patches = []
                                   


def gen_patch_cpu(x_i,w,h,start,section,q):
    patches = []
    for pixel in range(start,section+start): 
        x_pixel = pixel//(x_i.shape[0]-w+1)
        y_pixel = pixel%(x_i.shape[1]-h+1)              
        patches.append(gen_kernel(x_i,w,h,x_pixel,y_pixel,center=False))
    q.put(patches)

def preprocess_y(i):  
    i = classes.classimg(i,map=True)
    i_res = i.reshape((i.size,1))
    i_cat = label_binarize(i_res,[0,1,2,3,4,5])
    i_cat = i_cat.reshape((i.shape[:2]+(6,)))
    return i_cat

def remove_zero(a,b):
    if 1 not in a:
        return b
    else:
        return a

def preprocess_ys(ys,xs,labels=[0,1,2,3,4,5]):
    y_out = []
    null_rep = np.zeros(len(labels))
    null_rep[0] =1
    for i,j in zip(ys,xs):
        if i.shape != j.shape:
            i = misc.imresize(i,j.shape,interp='nearest')//255
        i = classes.classimg(i,map=True,labels=labels)
        i_res = i.reshape((i.size,1))
        i_cat = label_binarize(i_res,[0,1,2,3,4,5])
        i_cat = i_cat.reshape((i.shape[:2]+(6,))).astype(int)
        i_cat = i_cat[:,:,labels]
        #i_cat = np.apply_along_axis(remove_zero,2,i_cat,null_rep)
        y_out.append(i_cat)
    return y_out

def img_patches(xs,ys, patch_size, sync_seed=None, **kwards):
    np.random.seed(sync_seed)
    randint = np.random.randint(len(xs))
    x = xs[randint]
    y = ys[randint]
    w,h = x.shape[0],x.shape[1]
    rangew = (w - patch_size[0])
    rangeh = (h - patch_size[1])
    offsetw = np.random.randint(rangew)
    offseth = np.random.randint(rangeh)
    x = x[offsetw:offsetw+patch_size[0], offseth:offseth+patch_size[1], :]
    y = y[offsetw:offsetw+patch_size[0], offseth:offseth+patch_size[1], :]
    return x,y

def augment(x_patch,y_patch,patch_size):
    c_patch_size = x_patch.shape
    c_crop = (c_patch_size[0]-patch_size[0])//2
    #rotate
    #rotate = np.random.uniform(0,180)
    #y_patch = ndimage.rotate(y_patch, rotate, reshape=False)
    #x_patch = ndimage.rotate(x_patch, rotate, reshape=False)
    
    #crop to patch_size
    y_patch = y_patch[c_crop:c_crop+patch_size[0],c_crop:c_crop+patch_size[1]]
    x_patch = x_patch[c_crop:c_crop+patch_size[0],c_crop:c_crop+patch_size[1],:]

    #zoom
    
    #zf = np.random.uniform()*5
    #if zf > 4:
    #    zf = ((zf-4)*2)+1
    #    resize_patch = (int(patch_size[0]*zf),int(patch_size[1]*zf),y_patch.shape[2])
    #    y_patch1 = misc.imresize(y_patch[:,:,:3],resize_patch,interp="nearest")
    #    y_patch2 = misc.imresize(y_patch[:,:,3:],resize_patch,interp="nearest")
    #    y_patch = np.concatenate([y_patch1,y_patch2],axis=2)
    #    x_patch = misc.imresize(x_patch,resize_patch,interp="nearest")
    #    offset = (y_patch.shape[0]-patch_size[0])//2
    #    y_patch = y_patch[offset:offset+patch_size[0],offset:offset+patch_size[0],:]
    #    x_patch = x_patch[offset:offset+patch_size[0],offset:offset+patch_size[0],:]
    '''
    if np.random.random() < 0.5:
        y_patch = np.flip(y_patch, 0)
        x_patch = np.flip(x_patch, 0)
    if np.random.random() < 0.5:
        y_patch = np.flip(y_patch, 1)
        x_patch = np.flip(x_patch, 1)
    '''
    return x_patch,y_patch

def augment_x(x_patch,patch_size):
    c_patch_size = x_patch.shape
    c_crop = (c_patch_size[0]-patch_size[0])//2
    
    #crop to patch_size

    x_patch = x_patch[c_crop:c_crop+patch_size[0],c_crop:c_crop+patch_size[1],:]

    if np.random.random() < 0.5:
        x_patch = np.flip(x_patch, 0)
    if np.random.random() < 0.5:
        x_patch = np.flip(x_patch, 1)

    return x_patch   

def gen_aug_seq(x,y,patch_size,batch_size=64):
    constrained_patch = int(pow((patch_size[0]*patch_size[0]+patch_size[1]*patch_size[1]),1/2))
    l = -7
    while True:
        l +=7
        x_train = []
        y_train = []
        l = l%100
        for i in range(len(y)):
            for x_ind in range(0+l,y[i].shape[0]-constrained_patch,100):
                for y_ind in range(0+l,y[i].shape[1]-constrained_patch,100):
                    y_patch = y[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
                    x_patch = x[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
                    x_patch,y_patch = augment(x_patch,y_patch,patch_size)
                    y_train.append(y_patch)
                    x_train.append(x_patch)               
                    if len(y_train) == batch_size:
                        y_train = np.array(y_train)
                        x_train = np.array(x_train)
                        yield x_train, y_train
                        x_train = []
                        y_train = []

def gen_aug(x,y,patch_size,patch_size_out,batch_size=64,single_pixel=False,augment=False,labels=None,bal_val=1):
    to_bal = len(labels)<4 #TO CHANGE 
    attempt = 0
    balancer = False
    bal_min = int(bal_val*batch_size)
    bal_count = 0
    if not single_pixel and augment:
        constrained_patch = int(pow((patch_size[0]*patch_size[0]+patch_size[1]*patch_size[1]),1/2))
    else:
        constrained_patch = patch_size[0]
    part_size = batch_size//len(y)
    if part_size==0:
        part_size=1
    x_train = []
    y_train = []
    while True:
        for i in range(len(y)):
            options = int((y[i].shape[0]-constrained_patch)*(y[i].shape[1]-constrained_patch))
            sel_vals = np.random.permutation(options)
            part_amount = 0
            for val in sel_vals:
                x_ind,y_ind = get_xy(val,(y[i].shape[0]-constrained_patch))
                x_patch = x[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
                if single_pixel:
                    xoff = x_ind+(constrained_patch//2)
                    yoff = y_ind+(constrained_patch//2)
                    y_patch = y[i][xoff:xoff+1,yoff:yoff+1,:]
                    y_patch = np.reshape(y_patch,len(labels))
                    x_patch = augment_x(x_patch,patch_size)
                else:
                    y_patch = y[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
                    if augment:
                        x_patch,y_patch = augment(x_patch,y_patch,patch_size)
                    if patch_size_out:
                        out_x = (patch_size[0]//2)-(patch_size_out[0]//2)
                        out_y = (patch_size[1]//2)-(patch_size_out[1]//2)
                        y_patch = y_patch[out_x+1:-out_x,out_y+1:-out_y]
                if to_bal:
                    if not balancer and 1 in y_patch[:,:,1]:
                       bal_count += 1
                       if bal_count == bal_min:
                           balancer = True
                if (to_bal and not balancer
                       and bal_min-bal_count == batch_size-len(y_train)
                       and attempt < 5000):
                   attempt +=1
                else:
                    y_train.append(y_patch)
                    x_train.append(x_patch)
                    part_amount +=1
                    if part_amount == part_size:
                        break
            if len(y_train) == batch_size:
                #print()
                #print("attempt: "+str(attempt)+"  bal_count: "+str(bal_count))
                attempt = 0
                bal_count = 0
                balancer = False
                y_train = np.array(y_train)
                x_train = np.array(x_train)
                yield x_train, y_train
                x_train = []
                y_train = []

def gen_predef(path,batch_size=64):
    f_names = []
    for (dir_path, dir_names, file_names) in os.walk(path+'training_y'):
        f_names.extend(sorted(file_names))
        break
    i = 0 
    while True:        
        x_train = []
        y_train = []
        while len(y_train) < batch_size:
            i = i%len(f_names)
            rgb = misc.imread(path+'training_rgb/'+f_names[i], mode='RGB')/255.0
            d = misc.imread(path+'training_d/'+f_names[i])/255.0
            d = np.reshape(d,(d.shape[0],d.shape[1],1))
            y = misc.imread(path+'training_y/'+f_names[i],mode='RGB')/255.0
            y = preprocess_y(y) 
            x = np.concatenate((rgb,d[:,:,:1]),axis=2)
            x_train.append(x)
            y_train.append(y)
            i+=1
        ys = np.array(y_train)
        xs = np.array(x_train)
        yield xs, ys
        

def get_aug(x,y,patch_size,patch_size_out,batch_size=64):
    constrained_patch = int(pow((patch_size[0]*patch_size[0]+patch_size[1]*patch_size[1]),1/2))
    part_size = batch_size//len(y)
    if part_size==0:
        part_size=1
    x_train = []
    y_train = []

    for i in range(len(y)):
        options = int((y[i].shape[0]-constrained_patch)*(y[i].shape[1]-constrained_patch))
        sel_vals = np.random.permutation(options)
        for val in sel_vals:
            x_ind,y_ind = get_xy(val,(y[i].shape[0]-constrained_patch))
            y_patch = y[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
            x_patch = x[i][x_ind:x_ind+constrained_patch,y_ind:y_ind+constrained_patch,:]
            x_patch,y_patch = augment(x_patch,y_patch,patch_size)
            if patch_size_out:
                out_x = (patch_size[0]//2)-(patch_size_out[0]//2)
                out_y = (patch_size[1]//2)-(patch_size_out[1]//2)
                y_patch = y_patch[out_x+1:-out_x,out_y+1:-out_y]
            y_train.append(y_patch)
            x_train.append(x_patch)               
            if len(y_train) == part_size*(i+1):
                break
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    return x_train, y_train

def generator_train_patch(x,x_g,w,h,w_out=0,batch_size=64,categorical=False):
    if w_out==0:
        w_out = w
    offset = (w-w_out)//2
    while True:
        x_train = []
        y_train = []
        part_size = batch_size//len(x)
        if part_size==0:
            part_size=1
        for i in range(len(x)):
            x_img = x[i]
            y_img = classes.classimg(x_g[i],map=True)
            sel_vals = np.random.permutation((y_img.shape[0]-w)*(y_img.shape[1]-h))
            for val in sel_vals:
                x_ind,y_ind = get_xy(val,(y_img.shape[0]-w))
                y_train.append(gen_kernel(y_img,w_out,w_out,x_ind+offset,y_ind+offset,center=False))
                x_train.append(gen_kernel(x_img,w,h,x_ind,y_ind,center=False))               
                if len(y_train) == part_size*(i+1):
                    break
        y_train = np.array(y_train,dtype=int)

        if categorical:
            y_categorical = []
            for i in range(len(y_train)):
                y_t = y_train[i].reshape((y_train[i].size,1))
                #y_cat = np.zeros((y_t.size,6),dtype=int)
                y_cat = label_binarize(y_t,[0,1,2,3,4,5])
                y_cat = y_cat.reshape((y_train[i].shape[:2]+(6,)))
                y_categorical.append(y_cat)
            y_train = np.array(y_categorical)
        x_train = np.array(x_train)
        #return x_train, y_train
        yield x_train, y_train

def generator_train(x,x_g,w,h,CAT=None,batch_size=256):
    while True:
        ratio = 6
        part_size = batch_size//len(x)
        positive_max = part_size//ratio
        positive_max = positive_max if positive_max >= 1 else 1
        negative_max = part_size-positive_max
        balance = []
        x_train = []
        y_train = []

        mid = (w//2)
        pad = ((mid,mid),(mid,mid),(0,0))
        for i in range(len(x)):
            positive = 0
            negative = 0
            x_img = x[i]
            y_img = classes.classimg(x_g[i])
            x_img = np.pad(x_img,pad,'constant')
            sel_vals = np.random.permutation(y_img.shape[0]*y_img.shape[1])
            j = 0    
            for val in sel_vals:
                j +=1
                x_ind,y_ind = get_xy(val,y_img.shape[0])
                
                y_patch = gen_kernel(y_img,1,1,x_ind,y_ind)
                cat = y_patch[0,0,0]
                if cat not in balance or j>200:
                    balance.append(cat)
                else:
                   continue   
                y_train.append(y_patch)
                x_train.append(gen_kernel(x_img,w,h,x_ind,y_ind,center=False))               
                if len(balance) == 5:
                    balance=[]
                if len(y_train) == part_size*(i+1):
                    break

        y_train = np.array(y_train)
        if CAT:
            y_train = classes.multi(y_train,CAT)
        y_train = to_categorical(y_train,6) if not CAT else to_categorical(y_train,len(CAT)+1)
        x_train = np.array(x_train)

        yield x_train, y_train

    
def gen_data(x,x_g,shape,train=True,amount=1000):

    x_train = []
    y_train = []
    xoffset = int((shape[0])/2)
    yoffset = int((shape[1])/2)

    for i in range(len(x)):
        x_img = x[i]
        y_img = x_g[i]

        xs,ys = find_work_area(x_img,shape)      
        vals = xs*ys

        sel_vals = np.random.permutation(np.array(range(vals)))[:amount]
            
        for val in sel_vals:
            x_ind,y_ind = get_xy(val,xs)
            x_ind += xoffset
            y_ind += yoffset
            
            y_train.append(gen_kernel(y_img,(1,1),x_ind,y_ind))
            x_train.append(gen_kernel(x_img,shape,x_ind,y_ind))
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    
    return x_train,y_train
