import numpy as np
import aug
import math
ps = int(math.sqrt(2*150**2))+1
y = np.ones((ps,ps,1))
x =np.zeros((ps,ps,3))
a,b = aug.augment_patch(x,y,150)
print(a.shape)
print(b.shape)