import mynet
import mynet_card
import unet
import classnet
import resnet50
import carnet

def gen(input_shape,model,nclasses=6,p_out=None):
    if model=='mynet':
        if p_out:
            return mynet.build_out(input_shape,p_out,nclasses)
        else:
            return mynet.build(input_shape,nclasses)
    if model=='mynet_card':
        return mynet_card.build(input_shape,nclasses)    
    if model=='resnet50':
        return resnet50.build(input_shape,nclasses)
    if model=='unet':
        return unet.build(input_shape,nclasses)
    if model=='classnet':
        return classnet.build(input_shape,nclasses)
    if model=='carnet':
        return carnet.build(input_shape,nclasses)
