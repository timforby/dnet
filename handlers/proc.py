import numpy as np

class Process:
    @classmethod
    def setup(cls, xs, ys, s_l=None):
        """Sets up the labels for the network.
        #Arguments
            xs: (list of images) NxWxHxChannels
            ys: (list of images) NxWxHxChannels
            s_l: (list ints) Selected labels in range(len(classes)) to learn in network
        """

        cls.y, cls.y_classes = get_classes(ys)
        cls.max_label = np.max(cls.y_classes)+1
        cls.selected_labels = s_l
        cls.y_categorize = categorize_y()

    @staticmethod
    def get_classes(ys):
        """Sets classes found in ground truwdwth.
            Finds unique values in each image
            then unique values in whole list = cls.classes
        """
        ys_sc = []
        unique = np.array([])
        for i in range(len(ys)):
            ys_sc.append(to_single_channel(ys[i]))#flattening
            unique = np.concatenate(unique, np.unique(ys_sc[-1]))
        return ys_sc, np.unique(unique)

    @staticmethod
    def cat_imgs(xs,xs_,depth=1):
        return [np.concatenate(x,x_[:,:,:depth],axis=3) for x,x_ in zip(xs,xs_)]

    @staticmethod
    def label_bin(y, max_labels):
        _y = np.zeros((y.shape[0], max_labels),dtype=np.int)
        _y[np.arange(y.shape[0]),y] = 1
        return _y

    @staticmethod
    def to_single_channel(y):
        return y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])

    @staticmethod
    def to_three_channel(y):
        y = y.reshape(y.shape[0], y.shape[1], 3)
        return np.concatenate([y%2,(y//2)%2,(y//4)%2],axis=2)

    @static
    def categorize_img(y, y_classes, max_label, selected_labels):
        """Creates a categorized (1-hot) image on WxHxC where C is number of classes
            
        #Arguments
            y: (image) WxHx1 Must be flattened image with each value beign the class
                of that pixel
            y_classes: classes that exist within the ground truth
            max_label: maximum label found in y_classses
            selected_labels: user defined labels to select for training
        """
        _y = y.reshape((y.size,1))
        _y = label_bin(_y,max_label)
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
        _y = np.zeros((y.shape[:2]+(max_label,)))
        _y[:,:,y_classes] = y
        return _y

    @classmethod
    def categorize_y(cls):
        return [categorize_img(y, cls.y_classes, cls.max_label, cls.selected_labels) for y in cls.y] 

    @classmethod
    def uncategorize_imgs(cls, ys):
        return [uncategorize_img(y, cls.y_classes, cls.max_label, cls.selected_labels) for y in ys] 


    @classmethod
    def generate_patch(xs,ys,patch_size, batch_size=64, augment=False):
        x_train = []
        y_train = []

        while True:
            #Randomizes all possible locations into "patch_locations"
            patch_locations = []
            sum_locations = 0
            all_locations = []
            for y in ys:
                locations = int((y.shape[0]-patch_size[0])*(y.shape[1]-patch_size[1]))
                sum_locations += locations
                all_locations.append(locations)
                patch_locations.append(np.random.permutation(locations))
            patch_index = [0]*len(ys)
            
            i = 0
            while sum(patch_index) < sum_locations:
                #If patch_index has not reached all locations
                if patch_index[i] < all_locations[i]:
                    index=patch_locations[i][patch_index[i]]
                    x_patch = get_patch(xs[i], patch_size, index)
                    y_patch = get_patch(ys[i], patch_size, index)
                    x_train.append(x_patch)
                    y_train.append(y_patch)
                    patch_index[i] += 1
                i+=1
                i = i%len(y)

                if len(y_train) == batch_size:
                    y_train = np.array(y_train)
                    x_train = np.array(x_train)
                    yield x_train, y_train
                    x_train = []
                    y_train = []
        
        
    @staticmethod
    def get_patch(img, patch_size, index):
        x_val = index%(img.shape[0]-patch_size[0])
        y_val = index//(img.shape[0]-patch_size[0])
        return img[x_val:x_val+patch_size[0],y_val:y_val+patch_size[1],:]