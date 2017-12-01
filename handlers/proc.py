import numpy as np
from sklearn.preprocessing import label_binarize as lb

class Process:
	@classmethod
	def setup(cls, s_l, l):
		'''
		#Arguments
			s_l: Selected labels to learn in network
			l: labels that correspond to all labels in images when flattened to single channel.
				This is required because some images might not have all labels. Example = Red,Blue,Yellow = 1,4,3
		'''
		cls.selected_labels = s_l
		cls.max_labels = l
	
	@staticmethod
	def join_imgs(xs,ys,depth=1):
		return [np.concatenate(x,y[:,:,:depth],axis=3) for x,y in zip(xs,ys)]

	@classmethod
	def categorize_img(cls, y):
		y = cls.to_single_channel(y)
		_y = y.reshape((y.size,1))
		_sz = len(cls.max_labels)
		_y = lb(_y,range(_sz))
		_y = _y.reshape((y.shape[:2]+(_sz,))).astype(int)
		_y = _y[:,:,cls.selected_labels]
		return _y

	@classmethod
	def categorize_imgs(cls,ys):
		return [cls.categorize_img(y) for y in ys] 

	@classmethod	
	def to_single_channel(cls,y):
		y_res = y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])
		y_comp = np.copy(y_res)
		vals = [0]*len(cls.max_labels)	
		for l in cls.selected_labels:
			vals[l] = l
		for k, v in zip(cls.max_labels,cls.selected_labels):
			y_comp[y_res==k]= v

		return y_comp
	
	@classmethod
	def generate_patch(xs,ys,patch_size, batch_size=64, augment=False):
		x_train = []
		y_train = []

		while True:
			patch_locations = []
			max_options = 0
			for y in ys:
				options = int((y.shape[0]-patch_size[0])*(y.shape[1]-patch_size[1]))
				max_options = max(max_options,options)
				patch_locations.append(np.random.permutation(options))
			patch_index = [0]*len(ys)
			
			while
				index=patch_locations[i][patch_index[i]%len(patch_locations[i])]
				x_patch = getPatch(xs[i], index)
				y_patch = getPatch(ys[i], index)
