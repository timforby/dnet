import numpy as np
import load
from models.generator import Generator
from models.discriminator import Discriminator
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

def cat_batch(ys):
    result = np.zeros((ys.shape[:-1]+(1,)))
    for i in range(ys.shape[0]):
        y = ys[i,:,:,:]
        y = rgb_to_binary(y)
        #_1dy = y.reshape((y.size))
        #_2dy = np.zeros((_1dy.shape[0], 8),dtype=np.int)
        #_2dy[np.arange(_1dy.shape[0]),_1dy] = 1
        #result[i,:,:,:] = _2dy.reshape((y.shape[:2]+(8,)))      
        result[i,:,:,:] = y   
    return result

def binary_to_rgb(y):  
    y = y.reshape(y.shape[0], y.shape[1], 1)
    y = np.concatenate([y%2,(y//2)%2,(y//4)%2],axis=2)
    return y

def batch_b2rgb(ys):
    result = np.zeros((ys.shape[:-1]+(3,)))
    for i in range(ys.shape[0]):
        result[i,:,:,:] = binary_to_rgb(ys[i])
    return result



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
img_size = 64
batch_size = 32
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda:0")


netG = Generator(1).to(device)
netG.apply(weights_init)


netD = Discriminator(1).to(device)
netD.apply(weights_init)

criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
#optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))# SGD
optimizerD = optim.SGD(netD.parameters(), lr=0.01)
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print("Data setup")

rgb_wi_gt_data = loader(["../../data/small_potsdam/rgb","../../data/small_potsdam/y"],img_size,batch_size,transformations=[lambda x: x-load.get_mean("../../data/vaihingen/rgb"),rgb_to_binary])
data_wi_gt = rgb_wi_gt_data.generate_patch()



ones = torch.FloatTensor(batch_size).fill_(1).cuda()
zero = torch.FloatTensor(batch_size).fill_(0).cuda()

print("Running...")
counter = 1
for epoch in range(1000):
    errDiscriminitor = []
    errSegnet = []
    errGenerator = []
    for (x,y_bin) in data_wi_gt:
        x = x.transpose([0,3,1,2])
        y = y_bin.transpose([0,3,1,2])
        y = np.reshape(y,(batch_size,y[0].size,1,1))/7.0
        y = torch.from_numpy(y).cuda().float()
        #print("counter: "+str(counter))
        counter += 1


        real_rgb = torch.from_numpy(x).cuda().float()
        
        #DISCRIMINATOR
        netD.zero_grad()
        #forward with desired distribution
        disc_ouput = netD(real_rgb).view(-1)
        errD_real = criterion(disc_ouput, ones)
        errD_real.backward()
        D_x = disc_ouput.mean().item()


        #gen distrub
        noise = torch.FloatTensor(batch_size, 4096, 1, 1).normal_(0, 1).cuda()
        rgb_generator_input = torch.cat((y,noise),1).cuda()
        fake_rgb = netG(rgb_generator_input)

        #forward with generated distribution
        disc_ouput = netD(fake_rgb.detach()).view(-1)
        errD_fake = criterion(disc_ouput, zero)
        errD_fake.backward()
        D_G_z1 = disc_ouput.mean().item()


        #optimize
        errD = errD_real + errD_fake
        optimizerD.step()


        
        #GENERATOR -> DISCRIMINATOR

        netG.zero_grad()
        #forward with random + ground truth to generated distrib
        disc_ouput = netD(fake_rgb).squeeze()
        errG = criterion(disc_ouput, ones)
        errG.backward()
        D_G_z2 = disc_ouput.mean().item()
        #rgb_generator_opt.step()
        optimizerG.step()


        if counter % 50 == 0:
            print('[%d/%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, 1000, counter,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if counter % 5000 == 0:
            vutils.save_image(fake_rgb,'%s/%03d_epoch_%s.png' % (".", counter, "fake_rgb"),normalize=False)

            rgb_y = batch_b2rgb(y_bin).transpose([0,3,1,2])

            vutils.save_image(torch.from_numpy(rgb_y).float(),'%s/%03d_epoch_%s.png' % (".", counter, "gt"),normalize=False)
