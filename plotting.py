import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
import proc,classes

def calc_f1(x, y):
    x = x[:,:,0]+(2*x[:,:,1])+(4*x[:,:,2])
    y = y[:,:,0]+(2*y[:,:,1])+(4*y[:,:,2])
    Tp=Fp=Tn=Fn = 0
    labels = [2,4,3,6,7]
    for L in labels:
        # create result image where 1=label, 0=non label
        rb = (y==L).astype(int)
        # create ground image where 2=label, 0=non label
        gb = 2*((x==L).astype(int))
        # result: -1 = truepos, 1 = falsepos, -2= falseneg, 0 = trueneg
        res = rb-gb
        res = res.flatten()
        # result: 0 = falseneg, 1=truepos, 2=trueneg, 3=falsepos
        res +=2
        res = np.bincount(np.append(res,[0,1,2,3]))         
        Tp += (res[1]-1)//1
        Fn += (res[0]-1)//1
        Tn += (res[2]-1)//1
        Fp += (res[3]-1)//1
    Precision = Tp/(Tp+Fp)
    Recall = Tp/(Tp+Fn)
    F1 = 2*(Precision*Recall)/(Precision+Recall)
    return F1

class PlotLoss(keras.callbacks.Callback):
#class PlotLoss():

    def __init__(self, path, x, y, input, truth, patch_size, patch_out,plot_images=True,labels=None):
        self.input = input
        self.x = x
        self.y = y
        self.truth = truth
        self.labels = labels
        self.patch = patch_size
        self.patch_out = patch_out
        self.pi = plot_images
        path=path+'/plots'
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        return


    def on_epoch_end(self, epoch, logs = None):
        self.losses.append(logs.get('loss'))
        self.plot_training_curve()
        if self.pi:
            self.plot_validation(epoch)
            self.plot_f1_curve()
        return

    def plot_training_curve(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.losses, label='train')
        plt.legend(prop={'size': 15})
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        d = os.path.join(self.path, "costs.png")
        plt.savefig(d)
        plt.close('all')
        t = os.path.join(self.path, "log.txt")
        with open(t,"a+") as logger:
            logger.write(str(self.losses[-1])+"\n")
        
    def plot_f1_curve(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.accuracy, label='f1')
        plt.legend(prop={'size': 15})
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        d = os.path.join(self.path, "f1.png")
        plt.savefig(d)
        plt.close('all')

    # Plot a grid with 1 rows and 3 columns
    def plot_validation(self, epoch):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Epoch: ' + str(epoch), size=20)
        gs = gridspec.GridSpec(3, 3)
        for j in range(3):
            if j==2:
                dis,dts = proc.get_aug(self.x,self.y,self.patch, self.patch_out)
                di = dis[0]
                dt = dts[0]
            else:
                dis,dts = proc.get_aug(self.input,self.truth,self.patch, self.patch_out)
                di = dis[0]
                dt = dts[0]
            dp = self.model.predict(np.reshape(di,(1,)+di.shape))
            for h in range(3):
                i = (j*3)+h
                ax = plt.subplot(gs[i])
                if i % 3 == 0:
                    w = di
                    #w = di[:,:,np.array([0,1,4])]
                if i % 3 == 1:
                    pred = classes.declassimg2(np.argmax(dp[0],axis=2),map=True,cats=self.labels)
                    w = pred
                if i % 3 == 2:
                    truth = classes.declassimg2(np.argmax(dt,axis=2),map=True,cats=self.labels)
                    w = truth
                ax.imshow(w,
                          cmap=plt.cm.gist_yarg,
                          interpolation='nearest',
                          aspect='equal')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis('off')
                if i == 0:
                    ax.set_title("Input \n")
                elif i == 1:
                    ax.set_title("Prediction \n")
                elif i == 2:
                    ax.set_title("Truth \n")
                    if j < 2:
                        self.accuracy.append(calc_f1(pred,truth))
        gs.update(wspace=0)
        plt.savefig(os.path.join(self.path,str(epoch) + '_validation.png'))
        plt.close('all')
