from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers.merge import Add,Concatenate
from keras.layers.advanced_activations import LeakyReLU


def _batch_norm(input):
    '''Normalizes an input along channels
    '''
    return BatchNormalization(axis=3)(input)

def _conv2D(filters,kernal_size,dilation=1,padding='same',activation="relu",batch_norm="after",pooling="none",amount=1):
    '''Generates the basic residual block with convolution 2D
    '''
    def output(input):
        for i in range(amount):
            if batch_norm =="before":
                input = _batch_norm(input)
            input = Conv2D(filters,kernal_size,padding=padding,dilation_rate=dilation)(input)
            if batch_norm =="after":
                input = _batch_norm(input)
            if activation:
                input = Activation('relu')(input)
                #input = LeakyReLU(alpha=0.3)(input)
        if pooling=="max":
            input = MaxPooling2D(3)(input)
        elif pooling=="none":
            pass
        return input
    
        
    return output
  
def _conv2DTran(filters,kernal_size,dilation=1,padding='same',activation="relu",batch_norm="after",pooling="none",amount=1):
    '''Generates the basic residual block with convolution 2D
    '''
    def output(input):
        for i in range(amount):
            if batch_norm =="before":
                input = _batch_norm(input)
            input = Conv2DTranspose(filters,kernal_size,padding=padding,dilation_rate=dilation)(input)
            if batch_norm =="after":
                input = _batch_norm(input)
            if activation:
                input = Activation('relu')(input)
                #input = LeakyReLU(alpha=0.3)(input)
        if pooling=="max":
            input = MaxPooling2D(3)(input)
        elif pooling=="none":
            pass
        return input
    
        
    return output


def _group(amount, filter, kernal_size=3, dilation=1,batch_norm='after'):
    '''Generates a group of convolutional kernals that are merged
    '''
    outputs = []
    
    def output(input):   
        for i in range(amount):
            res = Conv2D(filter//amount,kernal_size,dilation_rate=dilation,padding="same")(input)
            outputs.append(res)
        
        return Concatenate()(outputs)
    
    return output
  
def _block(filter,cardinality,dilation=1,cv=True,batch_norm='after'):
    
    def output(input):
        btl = _conv2D(filter//2,1,batch_norm=batch_norm,dilation=dilation)(input)
        btl = _group(cardinality,filter//2,batch_norm=batch_norm,dilation=dilation)(btl)
        btl = _conv2D(filter,1,activation="",batch_norm=batch_norm,dilation=dilation)(btl)
        btl = Activation('relu')(btl)
        if cv:
            input = _conv2D(filter,1,activation="",batch_norm=batch_norm,dilation=dilation)(input)
        return Add()([btl,input])
    return output

def _blocker(amount,filter,cardinality=4,batch_norm='after',dilation=1):
    
    def output(input):
        input = _block(filter,cardinality,batch_norm=batch_norm,dilation=dilation)(input)
        for i in range(amount-1):
            input = _block(filter,cardinality,cv=False,batch_norm=batch_norm,dilation=dilation)(input)
        return input
    return output
    
def _split(input, amount=2):
    '''Splits a given input into and number 'amount' of different residual blocks
    '''
    outputs = []
    
    def output(input):
        return input
    
    for i in range(amount):
        outputs.append(output(input))
    
    return outputs
    
def _merge(*inputs):
    '''Merges all argument layers
    '''
    return Add()(list(inputs))
