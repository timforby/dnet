from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, UpSampling2D, Conv2DTranspose,Cropping2D
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Activation, Dense, Reshape, Lambda
from static.block import _conv2D,_conv2DTran,_blocker

def build3(img_size,nclasses=6):
    ps0,ps1,depth = img_size

    input = Input((ps0,ps1,depth))#200x200 ,150
    xinput = _conv2D(32,1,padding="valid")(input)#176, 135
    a = _conv2D(32,3,padding="valid")(input)#84
    a1 = _blocker(1,64)(a)#84
    a2 = MaxPooling2D((2,2))(a1)#88
    b = _conv2D(64,3,padding="valid")(a2)#84
    b1 = _blocker(1,128)(b)#84
    b2 = MaxPooling2D((2,2))(b1)#42
    c = _conv2D(128,3,padding="valid")(b2)#38
    c1 = _blocker(1,512)(c)#38
    c2 = MaxPooling2D((2,2))(c1)#19
    midl = _conv2D(1024,10,padding="valid")(c2)#10
    imid = _conv2DTran(512,10,padding="valid")(midl)#19
    ic2 = UpSampling2D((2,2))(imid)#38
    #ic2_a = Concatenate()([ic2,c1])#38
    ic2_c1 = Add()([ic2,c1])#38
    ic1 = _blocker(1,512)(ic2_c1)#38
    ic = _conv2DTran(128,5,padding="valid")(ic1)#42
    ib2 = UpSampling2D((2,2))(ic)#84
    #ib2_a = Concatenate()([ib2,b1])#84
    ib2_b1 = Add()([ib2,b1])#84
    ib1 = _blocker(1,128)(ib2_b1)#84
    ib = _conv2DTran(64,5,padding="valid")(ib1)#88
    ia2 = UpSampling2D((2,2))(ib)#176
    #ia1_a = Concatenate()([ia1,a])#176
    ia2_a1 = Add()([ia2,a1])#176
    ia1 = _blocker(1,64)(ia2_a1)
    ia = _conv2DTran(32,5,padding="valid")(ia1)#200
    ifinal_input = Add()([ia,xinput])
    ifinal = _conv2D(nclasses, 1,padding="valid")(ifinal_input)
    i = Reshape((ps0*ps1,nclasses))(ifinal)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)

def build(img_size,nclasses=6):
    ps0,ps1,depth = img_size

    input = Input((ps0,ps1,depth))

    a1 = _blocker(1,32,cardinality=8,dilation=2)(input)
    a2 = MaxPooling2D((2,2))(a1)

    b1 = _blocker(1,64,cardinality=16,dilation=2)(a2)
    b2 = MaxPooling2D((2,2))(b1)
    
    c1 = _blocker(1,128,cardinality=32,dilation=1)(b2)
    c2 = MaxPooling2D((2,2))(c1)
    
    midl = _conv2D(256,3,padding="valid")(c2)
    imid = _conv2DTran(128,3,padding="valid")(midl)
    
    ic2 = UpSampling2D((2,2))(imid)
    ic2_c1 = Concatenate()([ic2,c1])
    ic1 = _blocker(1,64,cardinality=32,dilation=1)(ic2_c1)

    ib2 = UpSampling2D((2,2))(ic1)
    ib2_b1 = Concatenate()([ib2,b1])
    ib1 = _blocker(1,32,cardinality=16,dilation=2)(ib2_b1)

    ia2 = UpSampling2D((2,2))(ib1)
    ia2_a1 = Concatenate()([ia2,a1])
    ia1 = _blocker(1,16,cardinality=8,dilation=2)(ia2_a1)

    ifinal_input = Concatenate()([ia1,input])
    ifinal = _conv2D(nclasses, 1,padding="valid")(ifinal_input)
    i = Reshape((ps0*ps1,nclasses))(ifinal)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)
    
def test(img_size):
    ps0,ps1,depth = img_size
    input = Input((ps0,ps1,depth))#200x200 ,150
    a = _conv2D(64,8,stride=2,padding="valid")(input)#176, 135
    a1 = MaxPooling2D((5,5))(a)#88
    b = _conv2D(128,4,padding="valid")(a1)#84
    return Model(inputs=input, outputs=b)
def build2(img_size,nclasses=6):
    ps0,ps1,depth = img_size

    input = Input((ps0,ps1,depth))#200x200 ,150
    a = _conv2D(64,10,padding="valid")(input)#176, 135
    a1 = MaxPooling2D((2,2))(a)#88
    b = _conv2D(64,3,padding="valid")(a1)#84
    b1 = _blocker(1,128)(b)#84
    b2 = MaxPooling2D((2,2))(b1)#42
    c = _conv2D(128,3,padding="valid")(b2)#38
    c1 = _blocker(1,512)(c)#38
    c2 = MaxPooling2D((2,2))(c1)#19
    midl = _conv2D(1024,3,padding="valid")(c2)#10
    imid = _conv2DTran(512,3,padding="valid")(midl)#19
    ic2 = UpSampling2D((2,2))(imid)#38
    ic2_a = Concatenate()([ic2,c1])#38
    ic1 = _blocker(1,512)(ic2_a)#38
    ic = _conv2DTran(128,3,padding="valid")(ic1)#42
    ib2 = UpSampling2D((2,2))(ic)#84
    ib2_a = Concatenate()([ib2,b1])#84
    ib1 = _blocker(1,128)(ib2_a)#84
    ib = _conv2DTran(64,3,padding="valid")(ib1)#88
    ia1 = UpSampling2D((2,2))(ib)#176
    ia1_a = Concatenate()([ia1,a])#176
    ia = _conv2DTran(nclasses,10,padding="valid")(ia1_a)#200
    i = Reshape((ps0*ps1,nclasses))(ia)
    iax = Activation("softmax")(i)
    out = Reshape((ps0,ps1,nclasses))(iax)

    return Model(inputs=input, outputs=out)

def build_out(img_size,p_out,nclasses=6):
    ps0,ps1,depth = img_size
    ps0_o,ps1_o = p_out

    input = Input((ps0,ps1,depth))#221
    crop = Cropping2D(cropping=(((ps0//2)-(ps0_o//2)+1,(ps0//2)-(ps0_o//2)),((ps1//2)-(ps1_o//2)+1,(ps1//2)-(ps1_o//2))))(input)
    a = _conv2D(64,10,padding="valid")(input)#212
    a1 = MaxPooling2D((2,2))(a)#106
    b = _conv2D(64,3,padding="valid")(a1)#104
    b1 = _blocker(1,128)(b)#104 
    b2 = MaxPooling2D((2,2))(b1)#52
    c = _conv2D(128,3,padding="valid")(b2)#50
    c1 = _blocker(1,512)(c)#50
    c2 = MaxPooling2D((2,2))(c1)#25
    midl = _conv2D(1024,3,padding="valid")(c2)#23
    imid = _conv2DTran(512,3,padding="valid")(midl)#25
    ic2 = UpSampling2D((2,2))(imid)#50
    ic2_a = Concatenate()([ic2,c1])#50
    ic1 = _blocker(1,512)(ic2_a)#50
    ia = _conv2DTran(128,3,padding="valid")(ic1)#52
    ia1_a = Concatenate()([ia,crop])
    #ia1 = UpSampling2D((2,2))(ia)#104
    #ia1_a = Concatenate()([ia1,b])#104
    ia1_b = _conv2D(nclasses,1,padding="valid")(ia1_a)
    i = Reshape((ps0_o*ps1_o,nclasses))(ia1_b)
    iax = Activation("softmax")(i)
    out = Reshape((ps0_o,ps1_o,nclasses))(iax)
    #m = Model(inputs=input,outputs=out)
    #print(m.summary())
    return Model(inputs=input, outputs=out)
