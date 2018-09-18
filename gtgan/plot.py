import torchvision.utils as vutils


class History():
    @classmethod
    def __init__(self, gen, disc, mepoch, mbatch):
        self.max_epoch = mepoch
        self.max_batch = mbatch
        self.gen = []
        self.gen_mean = []
        for g in gen:
            self.gen_mean.append(0)
            self.gen.append([])
        self.disc = []
        self.disc_mean = []
        for d in disc:
            self.disc_mean.append(0)
            self.disc.append([])

    @classmethod
    def add_gen(self, i, value):
        num = len(self.gen[i])
        self.gen_mean[i] = (self.gen_mean[i]*num+value)/(num+1)
        self.gen[i].append(value)

    @classmethod
    def add_disc(self, i, value):
        num = len(self.disc[i])
        self.disc_mean[i] = (self.disc_mean[i]*num+value)/(num+1)
        self.disc[i].append(value)

    @classmethod
    def print_stat(self, epoch, batch, i_d, i_g=0):
        print('[%d/%d][%d/%d] Loss_D%d: %.4f Loss_G%d: %.4f - %d' %
            (epoch, self.max_epoch, batch, self.max_batch, i_d, self.disc_mean[i_d], i_g, self.gen_mean[i_g], i_d))
    
    @staticmethod   
    def save_img(img, path, epoch, name):
        vutils.save_image(img,'%s/%03d_epoch_%s.png' % (path, epoch, name),normalize=False)