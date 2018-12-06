import load
import numpy as np

class loader:

    @classmethod
    def setup(cls, paths, patch_size, batch_size, augment=False):
        cls.orig_patch_size = patch_size
        if augment:
            patch_size = int(math.sqrt(2*cls.orig_patch_size**2))+1
            
        cls.patch_size = patch_size
        cls.batch_size = batch_size
        cls.augment = augment

        cls.data_details = load.load_data(paths[0], details_only=True)
        cls.data = load.load_data(paths[0])
        cls.mean = load.get_mean(paths[0])
        
    @classmethod        
    def generate_traverse(cls):
        #data details holds (shape0, shape1,path)
        #data traverse holds permute(0,1,2,...,shape0*shape1) minus patch size
        cls.data_traverse = []
        for x,y,path in cls.data_details:
            cls.data_traverse.append(np.random.permutation(np.arange((x-cls.patch_size) * (y-cls.patch_size))).tolist())

    @classmethod
    def generate_patch(cls):
        while True:
            cls.generate_traverse()
            #get number images, 
            num_data = len(cls.data_details)
            #get samples per images
            samples_per_image = cls.calc_samples_per_image(num_data)
            patches = []
            empty_image_index = []
            empty_image_trigger = False
            while len(empty_image_index) <= num_data:


                iterate_list = np.arange(num_data)
                if empty_image_trigger:
                    empty_image_trigger = False
                    iterate_list = np.setdiff1d(iterate_list,empty_image_index)
                    samples_per_image = cls.calc_samples_per_image(len(iterate_list))

                for i in np.random.permutation(iterate_list).tolist():
                    for s in range(samples_per_image):
                        patches.append(cls.get_patch(i, cls.data_traverse[i].pop()))
                        if not cls.data_traverse[i]:
                            empty_image_trigger = True
                            empty_image_index.append(i)

                        if len(patches) == cls.batch_size:
                            yield(np.array(patches))
                            patches = []

    @classmethod
    def calc_samples_per_image(cls, num_data):
        spi = cls.batch_size//num_data
        spi = 1 if spi < 1 else spi
        return spi

    @classmethod
    def get_patch(cls, image_index, index):
        try:
            img = cls.data[image_index]
        except:#NEEDS FIXING
            img = load.load_img(cls.data_details[image_index][2])
            cls.data[image_index] = img

        x_val = index%(img.shape[0]-cls.patch_size)
        y_val = index//(img.shape[0]-cls.patch_size)
        patch = img[x_val:x_val+cls.patch_size,y_val:y_val+cls.patch_size,:]
        return patch


lo = loader()
lo.setup(["C:\\Users\\abc\\Documents\\urbann\\data\\test\\rgb_ng"],150,32)
x = lo.generate_patch()
for i,p in enumerate(x):
    print(i)
    print(p.shape)
