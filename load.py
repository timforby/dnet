from scipy import misc
import os
import numpy as np
from random import uniform
from skimage import color


# Walk through path and get file names as strings
def get_filenames(path):
    f_names = []
    for (dir_path, dir_names, file_names) in os.walk(path):
        f_names.extend(sorted(file_names))
        break
    return f_names


def get_num(stri):
    num = stri.split('.')[0]
    return int(num)

# Load images in given directory
def load_imgs(fnames, start, end, dir_name,num_imgs,even):

    imgs = []
    count = 0
    for img in fnames:
        if not img.endswith("png"):
            continue
        try:
            if get_num(img) not in range(start,end):
                continue
        except ValueError:
            print(img+": not read :: could not parse int")
            continue
        if count >= num_imgs:
            break
        if even!=-1:
            if count % 2 != even:
                continue
        count += 1
        fullpath = os.path.join(dir_name, img)
        print(img, end=' ')
        img = misc.imread(fullpath)
        img = img/255.0
        if len(img.shape) ==2:
            img = np.reshape(img,(img.shape[0],img.shape[1],1))
        imgs.append(img)
    print()
    return imgs


def load_data(REAL_PATH, OTHER=None, PARENT_DIR=None, GROUND=True, num_imgs=9999,even=-1,start=0,end=9999):
    paths = [REAL_PATH+'y',REAL_PATH+'rgb',REAL_PATH+'d'] if GROUND else [REAL_PATH+'y_ng',REAL_PATH+'rgb_ng',REAL_PATH+'d_ng']

    if OTHER and not PARENT_DIR:
        paths.append(OTHER)

    if PARENT_DIR:
        if os.path.exists(PARENT_DIR):
            for (dir_path, dir_names, file_names) in os.walk(OTHER):
                for dir in sorted(dir_names):
                    paths.append(OTHER+dir)
                break   

    filename = []
    for path in paths:
        filename.append(get_filenames(path))
    
    filename = set(filename[0]).intersection(*filename)
    filename = sorted(filename)
    sets_imgs = []
    for path in paths:
        print("Loading: "+path)
        imgs = load_imgs(filename,start,end,path,num_imgs,even)
        sets_imgs.append(imgs)

    return sets_imgs


def load_all(REAL_PATH, comp=False):
    paths = [REAL_PATH+'y',REAL_PATH+'rgb',REAL_PATH+'d'] if not comp else [REAL_PATH+'y_ng',REAL_PATH+'rgb_ng',REAL_PATH+'d_ng']

    filename = []
    for path in paths:
        filename.append(get_filenames(path))

    filename = set(filename[0]).intersection(*filename)
    filename = sorted(filename)
    sets_imgs = []
    for path in paths:
        print("Loading: "+path)
        imgs = load_imgs(filename,0,9999,path,9999,-1)
        sets_imgs.append(imgs)

    return sets_imgs


def load_data_simp(paths, num_imgs=9999,even=-1,start=0,end=9999):

    sets_imgs = []
    for path in paths:
        print("Loading: "+path)
        imgs = load_imgs(get_filenames(path),start,end,path,num_imgs,even)
        sets_imgs.append(imgs)

    return sets_imgs




