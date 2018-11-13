import utils
import args as ar
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets as dset
import log
from GAN import GAN_Gen, GAN_Discriminator
from plot import History

def data_init(args):
    ## SEED ##    
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manual_seed)
    
    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader =torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=int(args.workers))

    ## TESTING DATA ##
    fixed_noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
    if args.cuda:
        fixed_noise = fixed_noise.cuda()

    return fixed_noise,dataloader

def init_discriminator(args):
    if args.model == "wgan":
        criterion = nn.L1Loss()
        use_sigmoid = False
    else:
        criterion = nn.BCELoss()
        use_sigmoid = True
    if args.cuda:
        criterion.cuda()
    disc = GAN_Discriminator(args.image_size, 3, args.ngpu, use_sigmoid)
    if args.inf != '':
        disc.load_state_dict(torch.load(args.inf+"/discriminator"))
    if args.model == "wgan":
        clip_weights(disc)
    if args.cuda:
        disc.cuda()

    if args.model == "wgan":
        optimizer = torch.optim.RMSprop
    else:
        optimizer = torch.optim.Adam
    opt = optimizer(disc.parameters(), lr=args.lr)

    return disc,opt,criterion

def init_unet(args):
    unet = UNet(n_classes=6)
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    criterion = nn.CELoss()
    if args.cuda:
        unet.cuda()
        criterion.cuda()
    return unet, optimizer, criterion

    
def init_generator(args, load=False, more_nz=0):
    gen = GAN_Gen(args.nz+more_nz, args.image_size, 3, args.ngpu)
    if load:
        gen.load_state_dict(torch.load(args.inf))
    if args.model == "wgan":
        optimizer = torch.optim.RMSprop
    else:
        optimizer = torch.optim.Adam
    opt = optimizer(gen.parameters(), lr=args.lr)

    if args.cuda:
        gen.cuda()
    return gen,opt


def clip_weights(model: nn.Module):
    for p in model.parameters():
        p.data.clamp_(-0.05, 0.05)

def train_batch(args, disc, gen, batch):
    discriminator = disc[0]
    optimizerD = disc[1]
    criterion = disc[2]

    generator = gen[0]
    optimizerG = gen[1]

    ####### TRAIN DESCRIMINATOR ######
    discriminator.zero_grad()

    #TRAIN WITH REAL
    real, ones, zero = batch
    real, _ = real
    if args.cuda:
        real = real.cuda()
        ones = ones.cuda()
        zero = zero.cuda()

    output = discriminator(real).view(-1, 1).squeeze(1)  # GAN_Discriminator train
    errD_real = criterion(output, ones[:output.shape[0]])
    errD_real.backward()

    #GENERATE FAKE
    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
    if args.cuda:
        noise = noise.cuda()
    fake = generator(noise)  # Generator train

    # TRAIN WITH FAKE
    output = discriminator(fake.detach()).view(-1, 1).squeeze(1)  # GAN_Discriminator train
    errD_fake = criterion(output, zero[:output.shape[0]])
    errD_fake.backward()

    errD = errD_real + errD_fake

    optimizerD.step()
    clip_weights(discriminator)

    ########## TRAIN GENERATOR ######
    generator.zero_grad()
    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
    if args.cuda:
        noise = noise.cuda()
    fake = generator(noise)  # Generator train
    output = discriminator(fake).view(-1, 1).squeeze(1)
    errG = criterion(output, ones)  # fake labels are real for generator cost
    errG.backward()
    optimizerG.step()

    return errD,errG

def train_batch_class(args, disc, gen, batch):
    discriminator = disc[0]
    optimizerD = disc[1]
    criterion = disc[2]

    generator = gen[0]
    optimizerG = gen[1]

    ####### TRAIN DESCRIMINATOR ######
    discriminator.zero_grad()

    #TRAIN WITH REAL
    real_rgb, real_gt, fake_gt = batch
    real_rgb, _ = real_rgb
    if args.cuda:
        real_rgb = real_rgb.cuda()
        real_gt = real_gt.cuda()
        fake_gt = fake_gt.cuda()

    output = discriminator(real_rgb)  # GAN_Discriminator train
    errD_real = criterion(output, real_gt)
    errD_real.backward()

    #GENERATE FAKE
    noise = torch.FloatTensor(args.batch_size, args.nz, fake_gt.shape[0], fake_gt.shape[1]).normal_(0, 1)
    if args.cuda:
        noise = noise.cuda()
    fake = generator(noise)  # Generator train

    # TRAIN WITH FAKE
    output = discriminator(fake.detach()).view(-1, 1).squeeze(1)  # GAN_Discriminator train
    errD_fake = criterion(output, zero[:output.shape[0]])
    errD_fake.backward()

    errD = errD_real + errD_fake

    optimizerD.step()
    clip_weights(discriminator)

    ########## TRAIN GENERATOR ######
    generator.zero_grad()
    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
    if args.cuda:
        noise = noise.cuda()
    fake = generator(noise)  # Generator train
    output = discriminator(fake).view(-1, 1).squeeze(1)
    errG = criterion(output, ones)  # fake labels are real for generator cost
    errG.backward()
    optimizerG.step()

    return errD,errG



def train(args, ds, gs, dataloader, test_noise):
   
    hist = History(gs, ds, args.niter, len(dataloader))

    ones = torch.FloatTensor(args.batch_size).fill_(1)
    zero = torch.FloatTensor(args.batch_size).fill_(0)
    for epoch in range(args.niter):
        for i, input in enumerate(dataloader, 0):
            eD,eG = train_batch(args, ds[0],gs[0],(input,ones,zero))
            hist.add_disc(0, eD)
            hist.add_gen(0, eG)
            hist.print_stat(epoch, i, 0)

            eD,eG = train_batch_class(args, ds[1],gs[0],(input,ones,zero))
            hist.add_disc(1, eD)
            hist.add_gen(0, eG)
            hist.print_stat(epoch, i, 1)

        if epoch % 2 ==0:
            fake = gs[0][0](test_noise)
            History.save_img(fake, args.outf, epoch, "d0")


        # do checkpointin
        if epoch % 200 ==0:
            torch.save(gs[0][0].state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
            torch.save(ds[0][0].state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))



def main():
    args = ar.getparams()
    test_noise,input = data_init(args)
    gen_rgb = init_generator(args,more_nz=101*101*3)
    gen_gt = init_generator(args,load=True)
    class_disc = init_unet(args)
    truth_disc = init_discriminator(args)
    log.store_in_text(args.outf, 'settings', args)
    train(args, [class_disc, truth_disc], [gen_rgb,gen_gt], input, test_noise)


if __name__ == '__main__':
    main()
