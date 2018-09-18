import torch
from torch import nn


class _transpose(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_size=64):
        super(_transpose, self).__init__()
        self.add_module('ct1', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))

    def forward(self, x):
        return super(_transpose, self).forward(x)


class GAN_Gen(nn.Module):
    def __init__(self, nz, imageSize, nc, ngpu):
        super(GAN_Gen, self).__init__()
        self.ngpu = ngpu
        nz = nz
        up = _transpose
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            up(nz, imageSize * 8, 3, 1, 0, output_size=3),
            nn.BatchNorm2d(imageSize * 8),
            nn.ReLU(True),
            # state size. (imageSize*8) x 3 x 3
            up(imageSize * 8, imageSize * 4, 4, 2, 1, output_size=6),
            nn.BatchNorm2d(imageSize * 4),
            nn.ReLU(True),
            # state size. (imageSize*4) x 6 x 6
            up(imageSize * 4, imageSize * 2, 4, 2, 1, output_size=12),
            nn.BatchNorm2d(imageSize * 2),
            nn.ReLU(True),
            # state size. (imageSize*2) x 12 x 12
            up(imageSize * 2, imageSize, 4, 2, 1, output_size=24),
            nn.BatchNorm2d(imageSize),
            nn.ReLU(True),
            # state size. (imageSize) x 24 x 24
            up(imageSize, imageSize//2, 4, 2, 1, output_size=48),
            nn.BatchNorm2d(imageSize//2),
            nn.ReLU(True),
            # state size. (imageSize) x 48 x 48
            up(imageSize//2, imageSize//4, 4, 2, 1, output_size=96),
            nn.BatchNorm2d(imageSize//4),
            nn.ReLU(True),
            # state size. (imageSize) x 96 x 96
            up(imageSize//4, nc, 6, 1, 0, output_size=101),
            nn.Tanh()
            # state size. (nc) x 101 x 101
        )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output


class GAN_Discriminator(nn.Module):
    def __init__(self, imageSize, nc, ngpu, use_sigmoid,out_channel=1):
        super(GAN_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 101 x 101
            nn.Conv2d(nc, imageSize//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize//4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 50 x 50
            nn.Conv2d(imageSize//4, imageSize//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize//2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x 25 x 25
            nn.Conv2d(imageSize//2, imageSize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (imageSize) x 12 x 12
            nn.Conv2d(imageSize, imageSize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (imageSize*2) x 6 x 6
            nn.Conv2d(imageSize * 2, imageSize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(imageSize * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 3 x 3
            nn.Conv2d(imageSize * 4, out_channel, 3, 1, 0, bias=False),
            # state size. (nimageSizedf*8) x 1 x 1
        )
        if use_sigmoid:
            self.main.add_module('sig', nn.Sigmoid())

    def forward(self, x, use_sigmoid=True):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output