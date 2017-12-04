from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose,Cropping2D
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Activation, Dense, Reshape, Lambda
from block import _conv2D,_conv2DTran,_blocker

def build(img_size,nclasses=6):
    ps0,ps1,depth = img_size

    input = Input((ps0,ps1,depth))#200x200 ,150
    a = _conv2D(32,15,padding="valid")(input)#176, 135
    a1 = MaxPooling2D((2,2))(a)#88
    b = _conv2D(32,5,padding="valid")(a1)#84
    b1 = _blocker(1,64,cardinality=16)(b)#84
    b2 = MaxPooling2D((2,2))(b1)#42
    c = _conv2D(64,5,padding="valid")(b2)#38
    c1 = _blocker(1,128,cardinality=32)(c)#38
    c2 = MaxPooling2D((2,2))(c1)#19
    midl = _conv2D(256,10,padding="valid")(c2)#10
    imid = _conv2DTran(128,10,padding="valid")(midl)#19
    ic2 = UpSampling2D((2,2))(imid)#38
    #ic2_a = Concatenate()([ic2,c1])#38
    ic2_a = Add()([ic2,c1])#38
    ic1 = _blocker(1,128,cardinality=32)(ic2_a)#38
    ic = _conv2DTran(64,5,padding="valid")(ic1)#42
    ib2 = UpSampling2D((2,2))(ic)#84
    #ib2_a = Concatenate()([ib2,b1])#84
    ib2_a = Add()([ib2,b1])#84
    ib1 = _blocker(1,64,cardinality=16)(ib2_a)#84
    ib = _conv2DTran(32,5,padding="valid")(ib1)#88
    ia1 = UpSampling2D((2,2))(ib)#176
    #ia1_a = Concatenate()([ia1,a])#176
    ia1_a = Add()([ia1,a])#176
    ia = _conv2DTran(nclasses,15,padding="valid")(ia1_a)#200
    i = Reshape((ps0*ps1,nclasses))(ia)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)
