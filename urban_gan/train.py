import os
os.chdir("C:\\Users\\abc\\Documents\\urbann\\dnet\\urban_gan")

import numpy as np
import load
from unet.unet import unet, half_unet
import proc
import torch
import torch.nn as nn
from random import randint

img_size = 150
batch_size = 32


rgb_generator = unet(8,3)
rgb_generator_opt = torch.optim.Adam(rgb_generator.parameters())
rgb_generator.cuda()

disc = half_unet(4,1)
disc_opt = torch.optim.Adam(disc.parameters())
disc_criterion = nn.BCELoss()
disc.cuda()

rgb_segment = unet(3,6)
rgb_segment_opt = torch.optim.Adam(rgb_generator.parameters())
rgb_segment_criterion = nn.CrossEntropyLoss()
rgb_segment.cuda()

rgb_no_gt = load.load_data("../../data/test/rgb_ng")
rgb_wi_gt = load.load_data("../../data/vaihingen/rgb")
rgb_gt = load.load_data("../../data/vaihingen/y")

rgb_wi_gt_data = proc.setup(x,y, None, None, img_size, batch_size)

data = rgb_wi_gt_data.generate_patch()

ones = torch.FloatTensor(batch_size).fill_(1).cuda()
zero = torch.FloatTensor(batch_size).fill_(0).cuda()

for epoch in range(1000):
    for x,y in data:
        fake_gt = torch.from_numpy(y)


        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()
        fake_rgb = rgb_generator(rgb_generator_input)

        rgb_no_gt_input = []
        for i in range(batch_size):
            img_num = randint(0,len(rgb_no_gt)-1)
            rgb_no_gt_input.append(proc.get_patch(rgb_no_gt[img_num],img_size,randint(0,rgb_no_gt[img_num].shape[0]*rgb_no_gt[img_num].shape[1])))
        rgb_no_gt_input = torch.from_numpy(np.array(rgb_no_gt_input)).cuda()

        rgb_wi_gt_input = torch.from_numpy(x).cuda()
        rgb_gt_input = torch.from_numpy(y).cuda()
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
        errS_fake = rgb_segment_criterion(rgb_segment_output, fake_gt)
        errS_fake.backward()

        errS = errS_real + errS_fake
        rgb_segment_opt.step()

        #GENERATOR -> SEGNET
        noise = torch.FloatTensor(batch_size, 2, img_size, img_size).normal_(0, 1)
        rgb_generator_input = torch.cat((fake_gt,noise),1).cuda()

        fake_rgb = rgb_generator(rgb_generator_input)
        rgb_segment_output = rgb_segment(fake_rgb)
        errG_s = rgb_segment_criterion(rgb_segment_output, fake_gt)
        errG_s.backward()

        errG = errG_d + errG_s
        rgb_generator_opt.step()

#generate ground truth
#generate noise

