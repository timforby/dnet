import torch
import torch.nn as nn
import torch.nn.functional as F

class multi_conv(nn.Module):
    def __init__(self, in_ch, out_ch, convs=2,k=3,s=1,p=1,d=1):
        super(multi_conv, self).__init__()
        self.convs = convs
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,k,stride=s,padding=p,dilation=d),
            nn.ReLU(inplace=True)
        )
        self.convs_m = nn.ModuleList()
        if convs > 1:
            for l in range(convs-1):
                self.convs_m.append(nn.Conv2d(out_ch, out_ch,k,stride=s,padding=p,dilation=d))
                self.convs_m.append(nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.conv1(x)
        for l in self.convs_m:
            x = l(x)
        return x


class vgg(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(vgg, self).__init__()
        self.mc1 = multi_conv(in_ch, 64)
        self.pixel_conv1 = multi_conv(64, 64, convs = 1, k=1, p=0)
        self.mxbasic = nn.MaxPool2d(3,stride=2,padding=1)
        self.mc2 = multi_conv(64, 128)
        self.pixel_conv2 = multi_conv(128, 128, convs = 1, k=1, p=0)
        self.mc3 = multi_conv(128, 256, convs=3)
        self.pixel_conv3 = multi_conv(256, 256, convs = 1, k=1, p=0)
        self.mc4 = multi_conv(256, 512, convs=3)
        self.mc5 = multi_conv(512, out_ch, convs=3, p=2,d=2)
        self.pixel_conv45 = multi_conv(out_ch, out_ch, convs = 1, k=1, p=0)
        self.mxp5 = nn.MaxPool2d(3,stride=1,padding=1)

    def forward(self, x):
        mc1_l = self.mc1(x)
        self.mc1_l_r = self.pixel_conv1(mc1_l)
        mc1_p = self.mxbasic(mc1_l)
        mc2_l = self.mc2(mc1_p)
        self.mc2_l_r = self.pixel_conv2(mc2_l)
        mc2_p = self.mxbasic(mc2_l)
        mc3_l = self.mc3(mc2_p)
        self.mc3_l_r = self.pixel_conv3(mc3_l)
        mc3_p = self.mxbasic(mc3_l)
        mc4_l = self.mc4(mc3_p)
        self.mc4_l_r = self.pixel_conv45(mc4_l)
        mc4_p = self.mxbasic(mc4_l)
        mc5_l = self.mc5(mc4_p)
        self.mc5_l_r = self.pixel_conv45(mc5_l)
        mc5_p = self.mxp5(mc5_l)

        return mc5_p


class context(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(context, self).__init__()
        self.conv_c = nn.Sequential(
            multi_conv(in_ch, 1024, convs=1, p=dilation, d=dilation),
            nn.Dropout(),
            multi_conv(1024, out_ch, convs=1, k=1, p=0)
            )

    def forward(self, x):
        x = self.conv_c(x)
        return x

class res_correction(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(res_correction, self).__init__()
        self.pixel_conv_red = multi_conv(in_ch, in_ch//4, convs = 1, k=1, p=0)
        self.conv_rc = multi_conv(in_ch//4, in_ch//4, convs=1)
        self.pixel_conv_up = multi_conv(in_ch//4, in_ch, convs=1, k=1, p=0)
        self.pixel_conv = multi_conv(in_ch, out_ch, convs = 1, k=1, p=0)

    def forward(self, x, y):
        x = x + y
        x_ = self.pixel_conv_red(x)
        x_ = self.conv_rc(x_)
        x_ = self.pixel_conv_up(x_)
        x = x_+x
        x = self.pixel_conv(x)
        return x
