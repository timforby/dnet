#from scasnet
#https://github.com/Yochengliu/ScasNet/blob/master/VGG_ScasNet.prototxt

from .scasnet_parts import *

class scasnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(scasnet, self).__init__()

        self.downsize = multi_conv(n_channels, n_channels, convs=1, k=(10,17), p=0)

        self.d = vgg(n_channels, 512)
        self.c1 = context(512, 512, 6)
        self.c2 = context(512, 512, 12)
        self.c3 = context(512, 512, 18)
        self.c4 = context(512, 512, 24)
        self.rc = res_correction(512,512)
        self.end = nn.Linear(512*26*45,n_channels)
               
    def forward(self, x):
        while x.shape[2:] != (405,720):
            x = self.downsize(x)
            #print(x.size())

        mc5_p = self.d(x)

        c1_l = self.c1(mc5_p)
        c2_l = self.c2(mc5_p)
        c3_l = self.c3(mc5_p)
        c4_l = self.c4(mc5_p)

        rc_34 = self.rc(c4_l, c3_l)
        rc_234 = self.rc(rc_34, c2_l)
        rc_1234 = self.rc(rc_234, c1_l)

        values = self.end(rc_1234.reshape(x.shape[0], -1))

        return values
