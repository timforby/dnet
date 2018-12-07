import load
import numpy as np

class loader:  
    def __init__(self, paths, patch_size, batch_size, transformations=[], augment=False):
        self.orig_patch_size = patch_size
        if augment:
            patch_size = int(math.sqrt(2*self.orig_patch_size**2))+1
            
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.augment = augment

        self.data_details = []
        self. data = []

        for path in paths:
            self.data_details.append(load.load_data(path, details_only=True))
            self.data.append(load.load_data(path))
        
        #ASSERT ALL SAME SIZE??? 
        
        for i,trans in enumerate(transformations):
            if trans:
                for j in range(len(self.data[i])):
                    self.data[i][j] = trans(self.data[i][j])
            
        #self.mean = load.get_mean(paths[0])
               
    def generate_traverse(self):
        #data details holds (shape0, shape1,path)
        #data traverse holds permute(0,1,2,...,shape0*shape1) minus patch size
        self.data_traverse = []
        for x,y,path in self.data_details[0]:
            self.data_traverse.append(np.random.permutation(np.arange((x-self.patch_size) * (y-self.patch_size))).tolist())

    def generate_patch(self):
        #get number images, 
        num_data = len(self.data_details[0])
        #get samples per images
        samples_per_image = self.calc_samples_per_image(num_data)
        patches_path = []
        batch_index = 0
        for j in range(len(self.data_details)):
            patches_path.append(np.zeros((self.batch_size,self.patch_size,self.patch_size,self.data[j][0].shape[2])))
        empty_image_index = []
        empty_image_trigger = False
        iterate_list = np.arange(num_data)
        while True:
            self.generate_traverse()
            while len(empty_image_index) <= num_data:
                if empty_image_trigger:
                    empty_image_trigger = False
                    iterate_list = np.setdiff1d(iterate_list,empty_image_index)
                    #print(iterate_list)
                    samples_per_image = self.calc_samples_per_image(len(iterate_list))
                for image_index in np.random.permutation(iterate_list).tolist():
                    for s in range(samples_per_image):
                        index = self.data_traverse[image_index].pop()
                        for path_index,patches in enumerate(patches_path):
                            patches[batch_index,:,:,:] = self.get_patch(path_index, image_index, index)
                        batch_index +=1
                        if batch_index == self.batch_size:
                            yield(patches_path)
                            batch_index = 0      
                        if len(self.data_traverse[image_index]) == 0:
                            empty_image_trigger = True
                            empty_image_index.append(image_index)
                            break

    def calc_samples_per_image(self, num_data):
        spi = self.batch_size//num_data
        spi = 1 if spi < 1 else spi
        return spi

    def get_patch(self, path_index, image_index, index):
        try:
            img = self.data[path_index][image_index]
        except:#NEEDS FIXING
            img = load.load_img(self.data_details[path_index][image_index][2])
            self.data[path_index][image_index] = img

        x_val = index%(img.shape[0]-self.patch_size)
        y_val = index//(img.shape[0]-self.patch_size)
        patch = img[x_val:x_val+self.patch_size,y_val:y_val+self.patch_size,:]
        return patch

'''
lo = loader()
#lo.setup(["C:\\Users\\abc\\Documents\\urbann\\data\\vaihingen\\rgb","C:\\Users\\abc\\Documents\\urbann\\data\\vaihingen\\y"],150,16)
lo.setup(["test_image","test_image"],519,8)
x = lo.generate_patch()
for i,p in enumerate(x):
    pass
    #print(i)
    #print(p[0].shape)
    #print(p[1].shape)
'''