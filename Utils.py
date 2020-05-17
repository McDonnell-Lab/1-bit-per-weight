import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def cutout(inputs,patch_size=18):
    
    '''T. Devries and G. W. Taylor. Improved regularization of convolutional neural networks with cutout. 
    http://arxiv.org/abs/1708.04552.'''
    
    #***************************************************************************************************
    #Definition of cutout to use random pixel values for CIFAR10 and CIFAR100 as in https://arxiv.org/abs/1802.08530
    #M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
    #ICLR, 2018
    #***************************************************************************************************

    img_h, img_w, img_c = inputs.shape
    Loc1 = np.random.randint(2-patch_size,img_h+1)
    Loc2 = np.random.randint(2-patch_size,img_w+1)
    top = np.maximum(0,Loc1-1)
    bottom = np.minimum(img_h-1,Loc1+patch_size-1-1)
    left = np.maximum(0,Loc2-1)
    right = np.minimum(img_w-1,Loc2+patch_size-1-1)
           
    if np.amax(inputs) <= 1.0:
        inputs = inputs*255.0
    inputs[top:bottom+1, left:right+1, :] = np.random.randint(0, 256, (bottom-top+1, right-left+1, img_c))

    #now fill padding with random integers. I set shifting to fill with constant = -1
    Mask=inputs<0
    c = np.count_nonzero(Mask)
    inputs[Mask] = np.random.randint(0,256,c)
    inputs = inputs/255.0

    return inputs

def GetDataGen(UseCutout): 
    
    #***************************************************************************************************
    #Definition of data augmentation for CIFAR10 and CIFAR100 as in https://arxiv.org/abs/1802.08530
    #M. D. McDonnell, Training wide residual networks for deployment using a single bit for each weight
    #ICLR, 2018
    #***************************************************************************************************
    
    #in 2018, keras docs for when the preprocessing function is applied are wrong. Its actually 
    #applied last. This is good. It means my random valued padding and cutout works correctly

    if UseCutout:
        datagen = ImageDataGenerator(preprocessing_function=cutout,
                                     width_shift_range=4/32,
                                     height_shift_range=4/32,
                                     horizontal_flip=True,
                                     fill_mode='constant',cval=-1) #note my cutout function changes cval=-1 to random 
                                                                 #ints so cutout is needed for this bit
    else:
        datagen = ImageDataGenerator(width_shift_range=4/32,
                                     height_shift_range=4/32,
                                     horizontal_flip=True,
                                     fill_mode='constant',cval=0)
        
    return datagen


#for implementing warm restarts in learning rate
class LR_WarmRestart(tensorflow.keras.callbacks.Callback):
    
    '''I. Loshchilov and F. Hutter. SGDR: stochastic gradient descent with restarts.
    http://arxiv.org/abs/1608.03983.'''
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.nbatch = nbatch
        self.currentEP=0.0
        self.startEP=1.0
        self.ThisBatch = 0.0
        self.lr_used=[]
        self.Tmult=0.0
        self.epochs_restart=epochs_restart
        
    def on_epoch_begin(self, epoch, logs={}):
        self.currentEP = self.currentEP+1.0
        self.ThisBatch = 0.0
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=self.currentEP+1.0
            K.set_value(self.model.optimizer.lr,self.initial_lr)
        print ('\n Start of Epoch Learning Rate = {:.6f}'.format(K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epochs, logs={}):
        print ('\n End of Epoch Learning Rate = {:.6f}'.format(self.lr_used[-1]))

        
    def on_batch_begin(self, batch, logs={}):
        
        pts = self.currentEP - self.startEP + self.ThisBatch/(self.nbatch-1.0)
        decay = 1.0+np.cos(pts/self.Tmult*np.pi)
        newlr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,newlr)
        
        #keep track of what we  use in this batch
        self.lr_used.append(K.get_value(self.model.optimizer.lr))
        self.ThisBatch = self.ThisBatch + 1.0
 
       
def plot_history(history):
    
    epochs = len(history['loss'])
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(range(1, epochs+1), history['loss'], label='loss')
    plt.plot(range(1, epochs+1), history['val_loss'],'g',label='val_loss')
    min_pos = np.argmin(history['val_loss'])
    min_value = history['val_loss'][min_pos]
    plt.scatter(min_pos+1, min_value, color='green', marker='*', label='min = ({}, {:.4f})'.format(min_pos+1, min_value))
    plt.ylabel('loss'); plt.xlabel('epoch'); plt.legend(); plt.grid()
    plt.subplot(122)
    plt.plot(range(1, epochs+1), history['acc'], label='acc')
    plt.plot(range(1, epochs+1), history['val_acc'],'g',label='val_acc')
    max_pos = np.argmax(history['val_acc'])
    max_value = history['val_acc'][max_pos]
    plt.scatter(max_pos+1, max_value, color='green', marker='*', label='max = ({}, {:.4f})'.format(max_pos+1, max_value))
    plt.ylabel('accuracy'); plt.xlabel('epoch'); plt.legend(); plt.grid()
    plt.show()
    