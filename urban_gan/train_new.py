import numpy as np
import load
from unet.unet import unet, half_unet
from new_loader import loader
import torch
import torch.nn as nn
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

img_size = 144
batch_size = 2

print("Net setup")
rgb_generator = unet(5,3)
rgb_generator_opt = torch.optim.Adam(rgb_generator.parameters())
rgb_generator.cuda()

disc = half_unet(3,1)
disc_opt = torch.optim.Adam(disc.parameters())
disc_criterion = nn.BCELoss()
disc.cuda()

rgb_segment = unet(3,6)
rgb_segment_opt = torch.optim.Adam(rgb_generator.parameters())
rgb_segment_criterion = nn.CrossEntropyLoss()
rgb_segment.cuda()

print("Data setup")
rgb_wi_gt_data = loader(["test_image/rgb","test_image/y"],img_size,batch_size,transformations=[None,rgb_to_binary])
#rgb_wi_gt_data = loader(["../../data/vaihingen/y","../../data/vaihingen/rgb"],img_size,batch_size,transformations=[None,rgb_to_binary])
data_wi_gt = rgb_wi_gt_data.generate_patch()


rgb_no_gt_data = loader(["test_image"],img_size,batch_size)
data_no_gt = rgb_no_gt_data.generate_patch()

fake_gt_data = loader(["test_image/y"],img_size,batch_size)
data_fake_gt = fake_gt_data.generate_patch()

ones = torch.FloatTensor(batch_size).fill_(1).cuda()
zero = torch.FloatTensor(batch_size).fill_(0).cuda()

print("Running...")
counter = 1
for epoch in range(1000):
    for (x,y),(z,),(fy,) in zip(data_wi_gt,data_no_gt,data_fake_gt):
        fy_cat = cat_batch(fy)
        y = y.transpose([0,3,1,2])
        x = x.transpose([0,3,1,2])
        z = z.transpose([0,3,1,2])
        fy = fy.transpose([0,3,1,2])
        fy_cat = fy_cat.transpose([0,3,1,2])
        
        print("counter: "+str(counter))
        counter += 1
        
        
        fake_gt = torch.from_numpy(fy).float()
        fake_gt_cat = torch.from_numpy(fy_cat.squeeze(1)).long().cuda()

        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()
        fake_rgb = rgb_generator(rgb_generator_input)

        rgb_no_gt_input = torch.from_numpy(z).cuda().float()
        rgb_wi_gt_input = torch.from_numpy(x).cuda().float()
        rgb_gt_input = torch.from_numpy(y.squeeze(1)).cuda().long()
        #DISCRIMINATOR
        disc.zero_grad()
        disc_ouput = disc(rgb_no_gt_input).view(-1, 1).squeeze(1) 
        errD_real = disc_criterion(disc_ouput, ones[:disc_ouput.shape[0]])
        errD_real.backward()

        disc_ouput = disc(fake_rgb.detach()).view(-1, 1).squeeze(1)
        errD_fake = disc_criterion(disc_ouput, zero[:disc_ouput.shape[0]])
        errD_fake.backward()

        errD = errD_real + errD_fake
        disc_opt.step()

        #GENERATOR -> DISCRIMINATOR
        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()

        rgb_generator.zero_grad()
        fake_rgb = rgb_generator(rgb_generator_input)
        disc_ouput = disc(fake_rgb).view(-1, 1).squeeze(1)
        errG_d = disc_criterion(disc_ouput, ones)
        errG_d.backward()
        #rgb_generator_opt.step()


        #SEGNET
        rgb_segment.zero_grad()
        rgb_segment_output = rgb_segment(rgb_wi_gt_input)
        errS_real = rgb_segment_criterion(rgb_segment_output, rgb_gt_input)
        errS_real.backward()

        rgb_segment_output = rgb_segment(fake_rgb.detach())
        errS_fake = rgb_segment_criterion(rgb_segment_output, fake_gt_cat)
        errS_fake.backward()

        errS = errS_real + errS_fake
        rgb_segment_opt.step()

        #GENERATOR -> SEGNET
        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()

        fake_rgb = rgb_generator(rgb_generator_input)
        rgb_segment_output = rgb_segment(fake_rgb)
        errG_s = rgb_segment_criterion(rgb_segment_output, fake_gt_cat)
        errG_s.backward()

        errG = errG_d + errG_s
        rgb_generator_opt.step()


        if counter % 500 == 0:
            vutils.save_image(fake_rgb,'%s/%03d_epoch_%s.png' % (".", counter, "fake_rgb"),normalize=False)
            #vutils.save_image(fake_gt,'%s/%03d_epoch_%s.png' % (".", counter, "fake_gt"),normalize=False)