from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Activation, Dense, Reshape, Lambda
from block import _conv2D,_conv2DTran,_blocker

def build2(img_size,nclasses=6):
    ps0,ps1,depth = img_size


    input = Input((ps0,ps1,depth))#200x200 ,150

    a = _conv2D(16,3,padding="valid")(input)
    a1 = MaxPooling2D((2,2))(a)
    b = _conv2D(32,3,padding="valid")(a1)
    b1 = MaxPooling2D((2,2))(b)
    c = _conv2D(64,3,padding="valid")(b1)
    c1 = MaxPooling2D((2,2))(c)
    d = _conv2D(128,3,padding="valid")(c1)
    d1 = MaxPooling2D((2,2))(d)
    mid = _conv2D(256,3,padding="valid")(d1)
    tmid = _conv2DTran(128,3,padding="valid")(mid)
    td1 = UpSampling2D((2,2))(tmid)
    td1_d = Add()([td1, d])
    td = _conv2DTran(64,3,padding="valid")(td1_d) 
    tc1 = UpSampling2D((2,2))(td)
    tc1_c = Add()([tc1, c])
    tc = _conv2DTran(32,3,padding="valid")(tc1_c)
    tb1 = UpSampling2D((2,2))(tc)
    tb1_b = Add()([tb1, b])
    tb = _conv2DTran(16,3,padding="valid")(tb1_b) 
    ta1 = UpSampling2D((2,2))(tb)
    ta1_a = Add()([ta1, a])
    ta = _conv2DTran(depth,3,padding="valid")(ta1_a) 
    
    all_lay = _conv2D(nclasses,1,padding="same")(ta)#84

    i = Reshape((ps0*ps1,nclasses))(all_lay)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)
    
def build(img_size,nclasses=6):
    ps0,ps1,depth = img_size


    input = Input((ps0,ps1,depth))

    a = _conv2D(16,3,padding="valid")(input)
    a1 = MaxPooling2D((2,2))(a)
    b = _conv2D(32,3,padding="valid",dilation=2)(a1)
    b1 = MaxPooling2D((2,2))(b)
    mid = _conv2D(64,3,padding="valid",dilation=2)(b1)
    tmid = _conv2DTran(32,5,padding="valid")(mid)
    tb1 = UpSampling2D((2,2))(tmid)
    #print(Model(inputs=input, outputs=tb1).summary())
    tb1_b = Concatenate()([tb1, b])
    tb = _conv2DTran(16,5,padding="valid")(tb1_b) 
    ta1 = UpSampling2D((2,2))(tb)
    ta1_a = Concatenate()([ta1, a])
    ta = _conv2DTran(depth,3,padding="valid")(ta1_a) 
    
    all_lay = _conv2D(nclasses,1,padding="same")(ta)

    i = Reshape((ps0*ps1,nclasses))(all_lay)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)
