import numpy as np
from . import aug
import math

class Process:
    @classmethod
    def setup(cls, xs, ys, xs_test,ys_test, patch_size, batch_size, mean, y_is_flat=False, s_l=None, one_hot=True):
        """Sets up the labels for the network.
        #Arguments
            xs: (list of images) NxWxHxChannels
            ys: (list of images) NxWxHxChannels
            s_l: (list ints) Selected labels in range(len(classes)) to learn in network
        """
        cls.mean = mean
        cls.input_shape = (patch_size,patch_size)+xs[0].shape[2:]
        cls.xs,cls.ys,cls.xs_test,cls.ys_test = xs,ys,xs_test,ys_test
        cls.patch_size, cls.batch_size = patch_size, batch_size
        cls.selected_labels = s_l
        cls.ys_flat, cls.y_classes = cls.get_classes(ys, y_is_flat)
        cls.max_label = int(np.max(cls.y_classes)+1)
        cls.y_categorize = cls.categorize_y(cls.ys_flat) if one_hot else cls.ys_flat

    @classmethod
    def setup_predict(cls, xs, patch_size, batch_size):
        cls.xs = xs
        cls.input_shape = (patch_size,patch_size,xs[0].shape[2:])
        cls.patch_size, cls.batch_size = patch_size, batch_size
        
    @staticmethod
    def get_classes(ys, y_is_flat):
        """Sets classes found in ground truwdwth.
            Finds unique values in each image
            then unique values in whole list = cls.classes
        """
        ys_sc = []
        unique = np.array([],dtype=np.int)
        for i in range(len(ys)):
            y = Process.to_single_channel(ys[i]) if not y_is_flat else ys[i]
            ys_sc.append(y)#flattening
            unique = np.concatenate([unique, np.unique(ys_sc[-1])])
        return ys_sc, np.unique(unique)

    @staticmethod
    def cat_imgs(xs,xs_,depth=1):
        return [np.concatenate([x,x_[:,:,:depth]],axis=2) for x,x_ in zip(xs,xs_)]

    @staticmethod
    def label_bin(y, max_labels):
        _y = np.zeros((y.shape[0], max_labels),dtype=np.int)
        _y[np.arange(y.shape[0]),y] = 1
        return _y

    @staticmethod
    def to_single_channel(y):
        y = y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])
        return y.astype(np.int)

    @staticmethod
    def to_three_channel(y):
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return np.concatenate([y%2,(y//2)%2,(y//4)%2],axis=2)

    @staticmethod
    def categorize_img(y, y_classes, max_label, selected_labels):
        """Creates a categorized (1-hot) image on WxHxC where C is number of classes
            
        #Arguments
            y: (image) WxHx1 Must be flattened image with each value beign the class
                of that pixel
            y_classes: classes that exist within the ground truth
            max_label: maximum label found in y_classses
            selected_labels: user defined labels to select for training
        """
        _y = y.reshape((y.size))
        _y = Process.label_bin(_y,max_label)
        _y = _y.reshape((y.shape[:2]+(max_label,))).astype(int)
        _y = _y[:,:,y_classes]
        if selected_labels:
            _y = _y[:,:,selected_labels]
        return _y

    @staticmethod
    def uncategorize_img(y, y_classes, max_label, selected_labels=None):
        """Creates a flat image WxHx1 from categorized image
            
        #Arguments
            y: (image) WxHxC Must be categorized image in onehot
        """
        
        if selected_labels:
            _y = np.zeros((y.shape[:2]+(len(y_classes),)))
            _y[:,:,selected_labels] = y
            y = _y.copy()
        _y = np.zeros((y.shape[:2]+(max_label+1,)))
        _y[:,:,y_classes] = y
        _y = np.argmax(_y,axis=2)
        _y = Process.to_three_channel(_y)
        return _y

    @classmethod
    def categorize_y(cls, ys):
        return [cls.categorize_img(y, cls.y_classes, cls.max_label, cls.selected_labels) for y in ys] 

    @classmethod
    def uncategorize_imgs(cls, ys):
        return [cls.uncategorize_img(y, cls.y_classes, cls.max_label, cls.selected_labels) for y in ys] 


    @classmethod
    def generate_patch(cls,augment=False):
        x_train = []
        y_train = []
        ps = cls.patch_size
        if augment:
            ps = int(math.sqrt(2*ps**2))+1
        while True:
            #Randomizes all possible locations into "patch_locations"
            patch_locations = []
            sum_locations = 0
            all_locations = []
            for y in cls.ys_flat:
                locations = int((y.shape[0]-ps)*(y.shape[1]-ps))
                sum_locations += locations
                all_locations.append(locations)
                patch_locations.append(np.random.permutation(locations))
            patch_index = [0]*len(cls.ys_flat)
            
            i = 0
            while sum(patch_index) < sum_locations:
                #If patch_index has not reached all locations
                if patch_index[i] < all_locations[i]:
                    index=patch_locations[i][patch_index[i]]
                    x_patch = cls.get_patch(cls.xs[i], ps, index)
                    y_patch = cls.get_patch(cls.y_categorize[i], ps, index)
                    if augment:
                        x_patch,y_patch = aug.augment_patch(x_patch,y_patch,cls.patch_size)
                    x_train.append(x_patch)
                    y_train.append(y_patch)
                    patch_index[i] += 1
                i+=1
                i = i%len(cls.ys_flat)

                if len(y_train) == cls.batch_size:
                    y_train = np.array(y_train)
                    x_train = np.array(x_train)-cls.mean
                    yield x_train, y_train
                    x_train = []
                    y_train = []
        
        
    @staticmethod
    def generate_predict_patch(image, patch_size, batch_size, mean):
        x_pred = []
        for r_indx in range(0, image.shape[0], patch_size):
            for c_indx in range(0, image.shape[1], patch_size):
                x_patch = image[r_indx:r_indx+patch_size,c_indx:c_indx+patch_size,:]
                x_pred.append(x_patch)
                if len(x_pred)==batch_size:
                    yield np.array(x_pred) - mean
                    x_pred = []
        yield np.array(x_pred)
        
    @staticmethod
    def get_patch(img, patch_size, index):
        x_val = index%(img.shape[0]-patch_size)
        y_val = index//(img.shape[0]-patch_size)
        return img[x_val:x_val+patch_size,y_val:y_val+patch_size,:]

    @classmethod
    def select_patch(cls,test=True,image_index=0,index=1500):
        if test:
            x = cls.get_patch(cls.xs_test[image_index],cls.patch_size,index)
            try:
                y = cls.get_patch(cls.ys_test[image_index],cls.patch_size,index)
            except:
                y = np.ones((cls.patch_size,cls.patch_size,3))
        else:
            x = cls.get_patch(cls.xs[image_index],cls.patch_size,index)
            y = cls.get_patch(cls.ys[image_index],cls.patch_size,index)
        return x,y
        
    @staticmethod
    def patch_factor_resize(image, patch_size):
        row = (1+image.shape[0]//patch_size)*patch_size
        col = (1+image.shape[1]//patch_size)*patch_size
        input = np.zeros((row,col)+(image.shape[2],))
        input[:image.shape[0],:image.shape[1],:] = image
        steps = input.shape[0]*input.shape[1]//(patch_size**2)
        return input,steps
