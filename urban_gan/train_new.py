import numpy as np
import load
from unet.unet import unet, half_unet
from unet.discriminator import Discriminator
from new_loader import loader
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import torchvision.utils as vutils


def rgb_to_binary(y):
    y = y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])
    y = np.expand_dims(y.astype(np.int),axis=2)
    #_1dy = y.reshape((y.size))
    #_2dy = np.zeros((_1dy.shape[0], 8),dtype=np.int)
    #_2dy[np.arange(_1dy.shape[0]),_1dy] = 1
    #y = _2dy.reshape((y.shape[:2]+(8,)))
    return y


def binary_to_rgb(y):  
    y = y.reshape(y.shape[0], y.shape[1], 1)
    y = np.concatenate([y%2,(y//2)%2,(y//4)%2],axis=2)
    return y

def batch_b2rgb(ys):
    result = np.zeros((ys.shape[:-1]+(3,)))
    for i in range(ys.shape[0]):
        result[i,:,:,:] = binary_to_rgb(ys[i])
    return result


def binary_to_rgb_torch(y):  
    y = torch.reshape(y,(1, y.shape[0], y.shape[1]))
    y = torch.cat([y%2,(y//2)%2,(y//4)%2],dim=0)
    return y

def batch_b2rgb_torch(ys):
    result = torch.zeros((ys.shape[0], 3, ys.shape[1], ys.shape[2]))
    for i in range(ys.shape[0]):
        result[i,:,:,:] = binary_to_rgb_torch(ys[i])
    return result

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

    
img_size = 128
batch_size = 32
lr = 0.0002
beta1 = 0.5



print("Net setup")
rgb_generator = unet(3,3)
rgb_generator_opt = torch.optim.Adam(rgb_generator.parameters(), lr=lr, betas=(beta1, 0.999))
rgb_generator.cuda()
rgb_generator.apply(weights_init)

disc = Discriminator(1)
disc_opt = torch.optim.SGD(disc.parameters(), lr=0.01)
disc_criterion = nn.BCELoss()
disc.cuda()
disc.apply(weights_init)


rgb_segment = unet(3,8)
rgb_segment_opt = torch.optim.Adam(rgb_segment.parameters())
rgb_segment_criterion = nn.CrossEntropyLoss()
rgb_segment.cuda()

print("Data setup")
rgb_wi_gt_data = loader(["../../data/small_potsdam/rgb","../../data/small_potsdam/y"],img_size,batch_size,transformations=[lambda x: x-load.get_mean("../../data/small_potsdam/rgb"),rgb_to_binary])
data_wi_gt = rgb_wi_gt_data.generate_patch()

#rgb_no_gt_data = loader(["../../data/test/rgb_ng"],img_size,batch_size,transformations=[])
#rgb_no_gt_data = loader(["../../data/test/rgb_ng"],img_size,batch_size,transformations=[lambda x: x-load.get_mean("../../data/test/rgb_ng")])
#data_no_gt = rgb_no_gt_data.generate_patch()
#fake_gt_data = loader(["../../data/vaihingen/y"],img_size,batch_size)
#data_fake_gt = fake_gt_data.generate_patch()

ones = torch.FloatTensor(batch_size).fill_(1).cuda()
zero = torch.FloatTensor(batch_size).fill_(0).cuda()
S_G_z1=0
D_G_z1=0
print("Running...")
counter =-1
for epoch in range(1000):
    errDiscriminitor = []
    errSegnet = []
    errGenerator = []
    for (x,y_bin) in data_wi_gt:
        x = x.transpose([0,3,1,2])
        y = y_bin.transpose([0,3,1,2])
        y_t = torch.from_numpy(y).cuda().float()
  
        counter += 1
        
        
        #fake_gt = torch.from_numpy(fy).float()
        #fake_gt_cat = torch.from_numpy(fy_cat.squeeze(1)).long().cuda
        #noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        #rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()
        #fake_rgb = rgb_generator(rgb_generator_input)

        #rgb_no_gt_input = torch.from_numpy(z).cuda().float()
        #rgb_wi_gt_input = torch.from_numpy(x).cuda().float()
        rgb_gt_input = torch.from_numpy(y.squeeze(1)).cuda().long()
        real_rgb = torch.from_numpy(x).cuda().float()
        real_rgb_clone = real_rgb.clone()
        
        #SEGMENT
        rgb_segment.zero_grad()
        rgb_segment_output = rgb_segment(real_rgb)
        errS_real = rgb_segment_criterion(rgb_segment_output, rgb_gt_input)
        errS_real.backward()

        rgb_segment_opt.step()
  
        #DISCRIMINATOR
        disc.zero_grad()
        #forward with desired distribution
        disc_ouput = disc(real_rgb_clone).view(-1)
        errD_real = disc_criterion(disc_ouput, ones)
        errD_real.backward()
        D_x = disc_ouput.mean().item()


         #GENERATOR -> DISCRIMINATOR
        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1).cuda()
        rgb_generator_input = torch.cat(((y_t+1)/4.0-1.0,noise),1).cuda()
        fake_rgb = rgb_generator(rgb_generator_input)
        fake_rgb_clone = fake_rgb.clone()


        #forward with generated distribution
        disc_ouput = disc(fake_rgb.detach()).view(-1)
        errD_fake = disc_criterion(disc_ouput, zero)
        errD_fake.backward()
        D_G_z1 = disc_ouput.mean().item()


        #optimize discriminator
        errD = errD_real + errD_fake
        disc_opt.step()


        rgb_generator.zero_grad()

        if counter % 5 == 0 and counter > 10000:
            #forward with random + ground truth to generated distrib
            rgb_segment_output = rgb_segment(fake_rgb_clone).squeeze()
            errG = rgb_segment_criterion(rgb_segment_output, rgb_gt_input)
            errG.backward()
            #rgb_generator_opt.step()

        else:
            #rgb_generator.zero_grad()
            #forward with random + ground truth to generated distrib
            disc_ouput = disc(fake_rgb).squeeze()
            errG = disc_criterion(disc_ouput, ones)
            errG.backward()
            D_G_z2 = disc_ouput.mean().item()

        rgb_generator_opt.step()



        if counter % 50 == 0:
            print('[%d/%d][%d]\tLoss_D: %.4f\tLoss_S: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, 1000, counter,
                     errD_real.item(), errS_real.item(), errG.item(), D_x, D_G_z1))
        if counter % 1000 == 0:
            vutils.save_image(real_rgb,'%s/%03d_epoch_%s.png' % (".", counter, "real_rgb"),normalize=False)
            vutils.save_image(fake_rgb,'%s/%03d_epoch_%s.png' % (".", counter, "fake_rgb"),normalize=False)
            rgb_y = batch_b2rgb(y_bin).transpose([0,3,1,2])
            vutils.save_image(torch.from_numpy(rgb_y).float(),'%s/%03d_epoch_%s.png' % (".", counter, "gt"),normalize=False)
            segment_y = batch_b2rgb_torch(torch.argmax(rgb_segment_output, dim=1))
            vutils.save_image(segment_y,'%s/%03d_epoch_%s.png' % (".", counter, "segment"),normalize=False)
