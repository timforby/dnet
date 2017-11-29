from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                        ZeroPadding2D, Conv2DTranspose,
                                        UpSampling2D)
from keras.layers.core import Dropout, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
################## KERAS ZOO MILA 


def build(img_shape, nclasses=6, l2_reg=0.,
               init='glorot_uniform', padding=100, dropout=True):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Input
    inputs = Input(img_shape, name='input')
    #padded = ZeroPadding2D(padding=(padding, padding), name='padded')(inputs)

    # Block 1
    conv1_1 = Conv2D(64, 3, padding='valid',
                            name='conv1_1', kernel_regularizer=l2(l2_reg))(inputs)
    conv1_2 = Conv2D(64, 3, padding='valid',
                            name='conv1_2', kernel_regularizer=l2(l2_reg))(conv1_1)
    conv1_2 = BatchNormalization(axis=3)(conv1_2)
    pool1 = MaxPooling2D((2, 2), (2, 2), name='pool1')(conv1_2)
    
    # Block 2
    conv2_1 = Conv2D(128, 3, padding='valid',
                            name='conv2_1', kernel_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Conv2D(128, 3, padding='valid',
                            name='conv2_2', kernel_regularizer=l2(l2_reg))(conv2_1)
    conv2_2 = BatchNormalization(axis=3)(conv2_2)
    pool2 = MaxPooling2D((2, 2), (2, 2), name='pool2')(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, 3, padding='valid',
                            name='conv3_1', kernel_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Conv2D(256, 3, padding='valid',
                            name='conv3_2', kernel_regularizer=l2(l2_reg))(conv3_1)
    conv3_2 = BatchNormalization(axis=3)(conv3_2)
    pool3 = MaxPooling2D((2, 2), (2, 2), name='pool3')(conv3_2)

    # Block 4
    conv4_1 = Conv2D(512, 3, padding='valid',
                            name='conv4_1', kernel_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Conv2D(512, 3, padding='valid',
                            name='conv4_2', kernel_regularizer=l2(l2_reg))(conv4_1)
    conv4_2 = BatchNormalization(axis=3)(conv4_2)
    if dropout:
        conv4_2 = Dropout(0.5, name='drop1')(conv4_2)
    pool4 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv4_2)

    # Block 5
    conv5_1 = Conv2D(1024, 3, padding='valid',
                            name='conv5_1', kernel_regularizer=l2(l2_reg))(pool4)
    conv5_2 = Conv2D(1024, 3, padding='valid',
                            name='conv5_2', kernel_regularizer=l2(l2_reg))(conv5_1)
    if dropout:
        conv5_2 = Dropout(0.5, name='drop2')(conv5_2)
    # pool5 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv5_2)
    conv5_2 = BatchNormalization(axis=3)(conv5_2)
    # Upsampling 1
    upconv4_1 = Conv2DTranspose(512, 3, padding='valid',
                              name='upconv4_1')(conv5_2)
    upconv4_2 = Conv2DTranspose(512, 3, padding='valid',
                              name='upconv4_2')(upconv4_1)
    upconv4_3 = UpSampling2D((2,2), name='upconv4_3')(upconv4_2)
    upconv4_3 = BatchNormalization(axis=3)(upconv4_3)
    #conv4_2_crop = CropLayer2D(upconv4, name='conv4_2_crop')(conv4_2)
    #upconv4_crop = CropLayer2D(upconv4, name='upconv4_crop')(upconv4)
    #Concat_4 = merge([conv4_2_crop, upconv4_crop], mode='concat', concat_axis=3, name='Concat_4')
    #Concat_4 = merge([conv4_2, upconv4_3], mode='concat', concat_axis=3, name='Concat_4')
    Concat_4 = Concatenate(name='Concat_4')([conv4_2,upconv4_3])
    upconv3_1 = Conv2DTranspose(256, 3,  padding='valid',
                            name='upconv3_1', kernel_regularizer=l2(l2_reg))(Concat_4)
    upconv3_2 = Conv2DTranspose(256, 3,  padding='valid',
                            name='upconv3_2', kernel_regularizer=l2(l2_reg))(upconv3_1)

    # Upsampling 2
    upconv3_3 = UpSampling2D((2,2), name='upconv3_3')(upconv3_2)
    upconv3_3 = BatchNormalization(axis=3)(upconv3_3)
    #conv3_2_crop = CropLayer2D(upconv3, name='conv3_2_crop')(conv3_2)
    #Concat_3 = merge([conv3_2, upconv3_3], mode='concat', name='Concat_3')
    Concat_3 = Concatenate(name='Concat_3')([conv3_2,upconv3_3])
    upconv2_1 = Conv2DTranspose(128, 3,  padding='valid',
                            name='upconv2_1', kernel_regularizer=l2(l2_reg))(Concat_3)
    upconv2_2 = Conv2DTranspose(128, 3,  padding='valid',
                            name='upconv2_2', kernel_regularizer=l2(l2_reg))(upconv2_1)

    # Upsampling 2
    upconv2_3 = UpSampling2D((2,2), name='upconv2_3')(upconv2_2)
    upconv2_3 = BatchNormalization(axis=3)(upconv2_3)
    #conv2_2_crop = CropLayer2D(upconv2, name='conv2_2_crop')(conv2_2)
    #Concat_2 = merge([conv2_2, upconv2_3], mode='concat', name='Concat_2')
    Concat_2 = Concatenate(name='Concat_2')([conv2_2,upconv2_3])
    upconv1_1 = Conv2DTranspose(64, 3,  padding='valid',
                            name='upconv1_1', kernel_regularizer=l2(l2_reg))(Concat_2)
    upconv1_2 = Conv2DTranspose(64, 3,  padding='valid',
                            name='upconv1_2', kernel_regularizer=l2(l2_reg))(upconv1_1)

    # Upsampling 2
    upconv1_3 = UpSampling2D((2,2), name='upconv1_3')(upconv1_2)
    upconv1_3 = BatchNormalization(axis=3)(upconv1_3)
    #conv1_2_crop = CropLayer2D(upconv1, name='conv1_2_crop')(conv1_2)
    #Concat_1 = merge([conv1_2, upconv1_3], mode='concat', name='Concat_1')
    Concat_1 = Concatenate(name='Concat_1')([conv1_2,upconv1_3])
    upconv0_1 = Conv2DTranspose(32, 3,  padding='valid',
                            name='upconv0_1', kernel_regularizer=l2(l2_reg))(Concat_1)
    upconv0_2 = Conv2DTranspose(32, 3,  padding='valid',
                            name='upconv0_2', kernel_regularizer=l2(l2_reg))(upconv0_1)
    
    Concat_0 = Concatenate(name='Concat_0')([inputs,upconv0_2])
    #Concat_0 = merge([inputs, upconv0_2], mode='concat', name='Concat_0')

    
    final_layer = Conv2D(nclasses,1,padding='valid',name='final_layer',kernel_regularizer=l2(l2_reg))(upconv0_2)
    final_layer = BatchNormalization(axis=3)(final_layer)
    # Crop
    #final_crop = CropLayer2D(inputs, name='final_crop')(conv10)
    # Softmax
    softmax_unet_0 = Reshape((img_shape[0]*img_shape[1],nclasses))(final_layer)
    softmax_unet_1 = Activation("softmax")(softmax_unet_0)
    softmax_unet = Reshape((img_shape[0],img_shape[1],nclasses))(softmax_unet_1)

    # Complete model
    model = Model(inputs=inputs, outputs=softmax_unet)


    return model
