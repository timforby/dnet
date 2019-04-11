#from scasnet
#https://github.com/Yochengliu/ScasNet/blob/master/VGG_ScasNet.prototxt

from .scasnet_parts import *

class scasnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(scasnet, self).__init__()

        self.d = vgg(n_channels, 512)
        self.c1 = context(512, 512, 6)
        self.c2 = context(512, 512, 12)
        self.c3 = context(512, 512, 18)
        self.c4 = context(512, 512, 24)
        self.rc = res_correction(512,512)
        self.rc4 = res_correction(512,256)
        self.rc3 = res_correction(256,256)
        self.fine_tune = multi_conv(256,256,convs=1)
        self.fine_tune_end = multi_conv(256,n_classes,convs=1, k=1)

        
        ## SETTING PARAMS IN GROUPS
        self.down = []
        self.cntx = []
        self.resc = []
        self.fine = []
        for n,m in self.named_modules():
            if '.' in n:
                continue
            if isinstance(m,context):
                for p in m.parameters():
                    self.cntx.append(p)
            elif isinstance(m,res_correction):
                for p in m.parameters():
                    self.resc.append(p)
            elif 'fine_tune' in n:
                for p in m.parameters():
                    self.fine.append(p)
            elif isinstance(m, vgg):
                for p in m.parameters():
                    self.down.append(p)

        
    def forward(self, x):
        mc5_p = self.d(x)

        c1_l = self.c1(mc5_p)
        c2_l = self.c2(mc5_p)
        c3_l = self.c3(mc5_p)
        c4_l = self.c4(mc5_p)

        rc_34 = self.rc(c4_l, c3_l)
        rc_234 = self.rc(rc_34, c2_l)
        rc_1234 = self.rc(rc_234, c1_l)

        rc_4 = self.rc(rc_1234, self.d.mc5_l_r)
        rc_4 = nn.functional.interpolate(rc_4, size=(50,50))
        rc_5 = self.rc4(rc_4, self.d.mc4_l_r)
        rc_5 = nn.functional.interpolate(rc_5, size=(100,100))
        rc_6 = self.rc3(rc_5, self.d.mc3_l_r)

        ft = self.fine_tune(rc_6)
        ft = self.fine_tune(ft)
        return nn.functional.interpolate(ft, size=(400,400))
