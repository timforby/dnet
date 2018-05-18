from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Activation, Dense, Reshape, Lambda
from block import _conv2D,_conv2DTran,_blocker

def build(img_size,nclasses=6):
    ps0,ps1,depth = img_size


    input = Input((ps0,ps1,depth))#200x200 ,150

    #FIRST BRANCH
    a = _conv2D(64,15,padding="valid")(input)#176, 135
    a1 = MaxPooling2D((2,2))(a)#88
    b = _conv2D(64,7,padding="valid")(a1)#84
    b2 = MaxPooling2D((2,2))(b)#42
    c = _conv2D(128,5,padding="valid")(b2)#38
    c2 = MaxPooling2D((2,2))(c)#19
    midl = _conv2D(1024,5,padding="valid")(c2)#10
    imid = _conv2DTran(512,5,padding="valid")(midl)#19
    ic2 = UpSampling2D((2,2))(imid)#38
    ic = _conv2DTran(128,5,padding="valid")(ic2)#42
    ib2 = UpSampling2D((2,2))(ic)#84
    ib = _conv2DTran(64,7,padding="valid")(ib2)#88
    ia1 = UpSampling2D((2,2))(ib)#176
    ia = _conv2DTran(1,15,padding="valid")(ia1)#200

    #SECOND BRANCH
    l_2_a = _conv2D(64,15,padding="valid")(input)#176, 135
    l_2_a1 = MaxPooling2D((2,2))(l_2_a)#88
    l_2_b = _conv2D(64,15,padding="valid")(l_2_a1)#84
    l_2_b2 = MaxPooling2D((2,2))(l_2_b)#42
    l_2_midl = _conv2D(1024,10,padding="valid")(l_2_b2)#10
    l_2_imid = _conv2DTran(512,10,padding="valid")(l_2_midl)#19
    l_2_ib2 = UpSampling2D((2,2))(l_2_imid)#84
    l_2_ib = _conv2DTran(64,15,padding="valid")(l_2_ib2)#88
    l_2_ia1 = UpSampling2D((2,2))(l_2_ib)#176
    l_2_ia = _conv2DTran(1,15,padding="valid")(l_2_ia1)#200


    #THIRD BRANCH
    l_3_a = _conv2D(64,15,padding="valid")(input)#176, 135
    l_3_a1 = MaxPooling2D((3,3))(l_3_a)#88
    l_3_b = _conv2D(64,5,padding="valid")(l_3_a1)#84
    l_3_b2 = MaxPooling2D((2,2))(l_3_b)#42
    l_3_midl = _conv2D(1024,10,padding="valid")(l_3_b2)#10
    l_3_imid = _conv2DTran(512,10,padding="valid")(l_3_midl)#19
    l_3_ib2 = UpSampling2D((2,2))(l_3_imid)#84
    l_3_ib = _conv2DTran(64,5,padding="valid")(l_3_ib2)#88
    l_3_ia1 = UpSampling2D((3,3))(l_3_ib)#176
    l_3_ia = _conv2DTran(1,15,padding="valid")(l_3_ia1)#200

    #FOURTH BRANCH
    l_4_a = _conv2D(3,15,padding="valid")(input)#176, 135
    l_4_a1 = AveragePooling2D((7,7))(l_4_a)#88
    l_4_b = _conv2D(64,7,padding="valid")(l_4_a1)#84
    l_4_b2 = MaxPooling2D((2,2))(l_4_b)#42
    l_4_midl = _conv2D(1024,3,padding="valid")(l_4_b2)#10
    l_4_imid = _conv2DTran(512,3,padding="valid")(l_4_midl)#19
    l_4_ib2 = UpSampling2D((2,2))(l_4_imid)#84
    l_4_ib = _conv2DTran(64,7,padding="valid")(l_4_ib2)#88
    l_4_ia1 = UpSampling2D((7,7))(l_4_ib)#176
    l_4_ia = _conv2DTran(1,15,padding="valid")(l_4_ia1)#200
    

    #FIFTH BRANCH
    l_5_a = _conv2D(64,3,padding="valid")(input)#176, 135
    l_5_a1 = MaxPooling2D((2,2))(l_5_a)#88
    l_5_midl = _conv2D(1024,10,padding="valid")(l_5_a1)#10
    l_5_imid = _conv2DTran(512,10,padding="valid")(l_5_midl)#19
    l_5_ia1 = UpSampling2D((2,2))(l_5_imid)#176
    l_5_ia = _conv2DTran(1,3,padding="valid")(l_5_ia1)#200

    #SIXTH BRANCH
    l_6_a = _conv2D(64,3,padding="same")(input)#176, 135
    l_6_b = _conv2D(1,15,padding="same")(l_6_a)#84
    
   
    all_lay = Concatenate()([ia,l_2_ia,l_3_ia,l_4_ia,l_5_ia,l_6_b])
    all_lay = _conv2D(nclasses,3,padding="same")(all_lay)#84

    i = Reshape((ps0*ps1,nclasses))(all_lay)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)
