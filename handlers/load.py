from scipy import misc
import os
import numpy as np

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
def load_imgs(dir_name, start, end, step):
    fnames = get_filenames(path)
    imgs = []
    count = 0
    for img in fnames:
        if not img.endswith("png"):
            continue
        try:
            if get_num(img) not in range(start,end,step):
                continue
        except ValueError:
            print(img+": not read :: could not parse int")
            continue
        count += 1
        fullpath = os.path.join(dir_name, img)
        img = misc.imread(fullpath)
        img = img/255.0
        if len(img.shape) ==2:
            img = np.reshape(img,(img.shape[0],img.shape[1],1))
        imgs.append(img)
    print()
    return imgs


def load_data(paths, start=0, end=999, step=1):
    """Load images into a list
    #Arguments
        paths: List of strings representing paths to folders containing
            images that must be named as numbers
        start,end,step: Refers to the number of name of images. Only loads
            images with in this range.
    """
    imgs = load_imgs(path,start,end,step)

    return imgs
    