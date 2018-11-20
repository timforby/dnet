


class loader:

    def setup(cls, paths, patch_size, batch_size, augment=False):
		cls.orig_patch_size = patch_size
		if augment:
            patch_size = int(math.sqrt(2*cls.orig_patch_size**2))+1
			
        cls.patch_size = patch_size
		cls.batch_size = batch_size
		cls.augment = augment

		cls.data_details = load.load_data(paths[0], dimensions_only=True)
		cls.mean = load.get_mean(paths[0])
		cls.generate_traverse()
		
	def generate_traverse(cls):
		cls.data_traverse = []
		cls.data_tracker = []
		for dim in cls.data_details:
			cls.data_traverse.append(np.random.permutation(np.arange(dim[0]-patch_size * dim[1]-patch_size)))
			cls.data_tracker.append(dim)
	
	def generate_patch(cls):

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
        
rgb_wi_gt_data = Process()
rgb_wi_gt_data.setup(rgb_wi_gt,rgb_gt, None, None, img_size, batch_size, load.get_mean("../../data/vaihingen/rgb"))
