import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras.callbacks as cb

class PlotLoss(cb.Callback):

    def __init__(self, path, gen):
        self.proc = gen
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
        
    # Plot a grid with 1 rows and 3 columns
    def plot_validation(self, epoch):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Epoch: ' + str(epoch), size=20)
        gs = gridspec.GridSpec(3, 3)
        for j in range(3):
            if j==2:
                di,dt = self.proc.select_patch()
            else:
                di,dt = self.proc.select_patch(test=False)
            dp = self.model.predict(np.reshape(di,(1,)+di.shape))
            for h in range(3):
                i = (j*3)+h
                ax = plt.subplot(gs[i])
                if i % 3 == 0:
                    w = di
                if i % 3 == 1:
                    pred = self.proc.uncategorize_imgs(np.argmax(dp,axis=3))
                    w = pred
                if i % 3 == 2:
                    w = dt
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
