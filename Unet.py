#_*_ coding:utf-8 _*_

from __future__ import print_function
#from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"]="15"
import numpy as np
import sys
import time
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
#from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,UpSampling2D,concatenate
#from tensorflow.keras.layers import add,concatenate
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
#from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,ModelCheckpoint




#lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
#early_stopper = EarlyStopping(min_delta=0.001, patience=10)
sys.setrecursionlimit(100000)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        #self.mean_squared_error = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        #self.val_mean_squared_error = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        #self.mean_squared_error['batch'].append(logs.get('mean_squared_error'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_mean_squared_error['batch'].append(logs.get('val_mean_squared_error'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        #self.mean_squared_error['epoch'].append(logs.get('mean_squared_error'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_mean_squared_error['epoch'].append(logs.get('val_mean_squared_error'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        #plt.plot(iters, self.mean_squared_error[loss_type], 'r', label='train mse')
        # loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            #plt.plot(iters, self.val_mean_squared_error[loss_type], 'b', label='val mse')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'g', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig('loss.tif',dpi=400)
        plt.show()

def load_data():

  #-------train part------
  X_train=np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_train/DMR_npy.npy")
  X_train = X_train.reshape(len(X_train),256,256,1)
  X_train = X_train / 255
  Y_train=np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_train/MR_npy.npy")
  Y_train = Y_train.reshape(len(Y_train),256,256,1)
  Y_train = Y_train / 255
  #-------test part-------
  X_test=np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_test/p45/DMR_npy.npy")
  X_test = X_test.reshape(len(X_test),256,256,1)
  X_test = X_test / 255
  #Y_test=np.load('/home/liwen/Desktop/head_mr-ct_paper/data/4/ct.npy')#()

  '''X_train/=X_train.max()
  mean = X_train.mean()# (0)[np.newaxis,:]  # mean for data centering
  std = np.std(X_train)  # std for data normalization
  X_train -= mean
  X_train /= std'''
  '''X_train /= 255
  mean = X_train.mean()
  std = np.std(X_train)
  X_train -= mean
  X_train /= std

  Y_train /= 255
  mean = Y_train.mean()
  std = np.std(Y_train)
  Y_train -= mean
  Y_train /= std

  X_test /= 255
  mean = X_test.mean()
  std = np.std(X_test)
  X_test -= mean
  X_test /= std'''


  #X_train=np.rollaxis(X_train,2,0)
  #X_train=X_train.reshape(3690,512,512,1)

  #Y_train=np.rollaxis(Y_train,2,0)
  #Y_train=Y_train.reshape(3690,512,512,1)

  #X_test=np.rollaxis(X_test,2,0)
  #X_test=X_test.reshape(30,512,512,1)


  #print ('after_reshape_X_train.shape:', X_train.shape)
  #print ('after_reshape_Y_train.shape:', Y_train.shape)
  #print ('after_reshape_X_test.shape:', X_test.shape)
  return (X_train,Y_train,X_test)
  #return (X_test)

print('-'*60)

def _bn_relu(input):

    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

def _bn_relu_conv(**conv_params):

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):

    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])

print('-'*60)

def Unet(img_rows,img_cols,img_channels,num_classes=None):
    inputs=Input(shape=(img_rows, img_cols, img_channels))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)


    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge6 = concatenate([conv3,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv2,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv1,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

   
    #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv8)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mse', metrics = ['mae'])

    return model

if __name__=='__main__':

    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS

    ROW_AXIS = 1
    COL_AXIS = 2
    CHANNEL_AXIS = 3
 
    batch_size = 16
    epochs = 200
    img_rows, img_cols = 256,256
    img_channels = 1
    
    X_train,Y_train,X_test = load_data()
    #X_test = load_data()
   
    model=Unet(img_rows, img_cols, img_channels)

    #plot_model(model, to_file='model.png',show_shapes=True)
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    start_time=time.clock()
    a=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    history = LossHistory()
    #model.load_weights('unet.hdf5')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.25,shuffle=True,callbacks=[model_checkpoint,history])
    end_time=time.clock()
    print(a)
    print ('train_time_is:',(end_time-start_time))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    history.loss_plot('epoch')
    print(history.losses['epoch'])
    print(history.val_loss['epoch'])
    losses=history.losses['epoch']
    val_loss=history.val_loss['epoch']
    np.save('loss_log.npy',losses)
    np.save('val_loss_log.npy',val_loss)

    #plt.savefig('result.tif',*,dpi=600)
    #model.save_weights('models/spine.hdf5')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting on test data...')
    print('-'*30)
    start_time=time.clock()
    b=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    prediction_results = model.predict(X_test,batch_size=4, verbose=1)
    #prediction_results = prediction_results.reshape(30,512,512)
    end_time=time.clock()
    print ('test_time_is:',(end_time-start_time))
    print(b)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    np.save('result_45.npy', prediction_results)
    imgs = np.load('result_45.npy')
    #imgs = imgs.reshape(30,512,512)
    for i in range(imgs.shape[0]):
      img = imgs[i]
      img = array_to_img(img)
      img.save("singal_picture/%d.png"%(i))

    print ('Done......')
