#dnet


class DNET(nn.Module):
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