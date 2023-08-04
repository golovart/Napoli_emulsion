import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, re, gc
import h5py, skimage
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.layers import ZeroPadding3D, Conv3D, MaxPooling3D, Concatenate, Add
from keras import optimizers
from keras.models import Model, load_model
from keras.utils import multi_gpu_model
from keras.constraints import MaxNorm
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import HDF5Matrix, to_categorical
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
K.set_image_data_format('channels_last')


class call_roc_hist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_aucs = []
        #self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        #self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.validation_data[0])
        scoroc = roc_auc_score(self.validation_data[1], y_pred)
        self.val_aucs.append(scoroc)
        #print('\n',epoch,'\troc_auc:',scoroc,'\n')
        return
    
def new_input_shape(model, input_shape):    
    newInput = Input(shape=input_shape)
    newOutputs = model(newInput)
    new_model = Model(newInput, newOutputs, name=model.name+'inp_chng')
    new_model.compile(optimizer=model.optimizer, loss=model.loss)
    #new_model.summary()
    return new_model


### CNN building blocks ###
def swish(x):
    """
    x*sigmoid(x)
    """
    return (K.sigmoid(x) * x)


def identity_block(X, f, filters, stage, block):
    """    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_D_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_D, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    #X = Conv3D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name = conv_name_base + '2a')(X)
    X = Conv3D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '2a')(X)
    X = Activation('swish')(X)
    
    # Second component of main path (≈3 lines)
    #X = Conv3D(filters=F2, kernel_size= f, padding='same', kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name= conv_name_base+'2b')(X)
    X = Conv3D(filters=F2, kernel_size= f, padding='same', kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name= conv_name_base+'2b')(X)
    X = BatchNormalization(axis=3, name= bn_name_base+'2b')(X)
    X = Activation('swish')(X)

    # Third component of main path (≈2 lines)
    #X = Conv3D(filters=F3, kernel_size=1, padding='valid', kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name=conv_name_base+'2c')(X)
    X = Conv3D(filters=F3, kernel_size=1, padding='same', kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a SWISH activation (≈2 lines)
    X = Add()([X, X_shortcut])#X+X_shortcut
    X = Activation('swish')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_D_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_D, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    #X = Conv3D(F1, 1, strides = s, kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name = conv_name_base + '2a')(X)
    X = Conv3D(F1, 1, strides = s, kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = -1, name = bn_name_base + '2a')(X)
    X = Activation('swish')(X)
    
    # Second component of main path (≈3 lines)
    #X = Conv3D(F2, f, kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), padding='same', name=conv_name_base+'2b')(X)
    X = Conv3D(F2, f, kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), padding='same', name=conv_name_base+'2b')(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2b')(X)
    X = Activation('swish')(X)

    # Third component of main path (≈2 lines)
    #X = Conv3D(F3, 1, kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name=conv_name_base+'2c')(X)
    X = Conv3D(F3, 1, kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), padding='same', name=conv_name_base+'2c')(X)
    X = BatchNormalization(axis=-1, name=bn_name_base+'2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    #X_shortcut = Conv3D(F3, 1, strides=s, kernel_regularizer=regularizers.l2(1e-3), kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), name=conv_name_base+'1')(X_shortcut)
    X_shortcut = Conv3D(F3, 3, strides=s, kernel_constraint=MaxNorm(4., axis=[0,1,2,3]), padding='same', name=conv_name_base+'1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=-1, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a SWISH activation (≈2 lines)
    X = Add()([X, X_shortcut]) #X+X_shortcut
    X = Activation('swish')(X)
    
    X = Dropout(rate=0.4, name='drop'+str(stage)+block)(X)
    
    return X






### BATCH TRAINING ###

class Batch_data_generator(keras.utils.Sequence):
    #images and labels are expected to be HDF5 datasets
    def __init__(self, dataset=None, run_type='train', batch_size=256, n_batch=None, shuff=True):
        assert not (dataset is None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.rtype = run_type
        self.batch_ids = np.array(list(self.dataset[self.rtype].keys()), dtype=int)
        if isinstance(n_batch,int): self.batch_ids = np.random.permutation(self.batch_ids)[:n_batch]
        self.shuffle = shuff

    def __len__(self):
        return len(self.batch_ids)
    
    def on_epoch_end(self):
        # to shuffle indices each epoch
        if self.shuffle:
            np.random.shuffle(self.batch_ids)

    def __getitem__(self, idx):
        idx = self.batch_ids[idx]
        batch_x = self.dataset[self.rtype+'/'+str(idx)+'/images'][...]
        batch_x = batch_x/256.
        batch_y = self.dataset[self.rtype+'/'+str(idx)+'/labels'][...]
        return batch_x[...,np.newaxis], batch_y
    
class Batch_data_augmentator(keras.utils.Sequence):
    #images and labels are expected to be HDF5 datasets
    def __init__(self, dataset=None, run_type='train', batch_size=256, n_batch=None, shuff=True, angles_list=None):
        assert not (dataset is None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.rtype = run_type
        self.batch_ids = np.array(list(self.dataset[self.rtype].keys()), dtype=int)
        if isinstance(n_batch,int): self.batch_ids = np.random.permutation(self.batch_ids)[:n_batch]
        self.shuffle = shuff
        #self.aug_train = ImageDataGenerator(rotation_range=90, fill_mode='reflect', dtype=float)
        self.angles = angles_list

    def __len__(self):
        return len(self.batch_ids)
    
    def on_epoch_end(self):
        # to shuffle indices each epoch
        if self.shuffle:
            np.random.shuffle(self.batch_ids)

    def __getitem__(self, idx):
        idx = self.batch_ids[idx]
        batch_x = self.dataset[self.rtype+'/'+str(idx)+'/images'][...]
        batch_x = batch_x/256.
        batch_y = self.dataset[self.rtype+'/'+str(idx)+'/labels'][...]
        #params = self.aug_train.get_random_transform(batch_x.shape[1:])
        rot_angles = np.random.choice(self.angles, batch_x.shape[0])
        for i in range(batch_x.shape[0]):
            #batch_x[i] = self.aug_train.apply_transform(batch_x[i], params)
            for i_pol in range(batch_x.shape[-1]):
                batch_x[i,...,i_pol] = skimage.transform.rotate(batch_x[i,...,i_pol], angle=rot_angles[i], mode='symmetric', order=3)
        return batch_x[...,np.newaxis], batch_y
    
    
# numpy-array approach
class Batch_data_generator_np(keras.utils.Sequence):
    #images and labels are expected to be HDF5 datasets
    def __init__(self, images, labels, batch_size=256, shuff=True, multiclass=True, n_cl=5):
        self.images, self.labels = images, labels
        self.batch_size = batch_size
        self.indices = np.arange(self.images.shape[0])
        self.batch_ids = np.arange(np.ceil(self.images.shape[0] / float(self.batch_size)), dtype=int)
        self.shuffle = shuff
        self.multiclass = multiclass
        self.num_class = n_cl

    def __len__(self):
        return int(np.ceil(self.images.shape[0] / float(self.batch_size)))
    
    def on_epoch_end(self):
        # to shuffle indices each epoch
        if self.shuffle:
            np.random.shuffle(self.batch_ids)

    def __getitem__(self, idx):
        idx = self.batch_ids[idx]
        load_ids = list(self.indices[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = self.images[load_ids]
        #batch_x = batch_x/256.
        batch_y = self.labels[load_ids]
        if self.multiclass: batch_y = to_categorical(batch_y, num_classes=self.num_class)
        return batch_x[...,np.newaxis], batch_y
    
class Batch_data_augmentator_np(keras.utils.Sequence):
    #images and labels are expected to be HDF5 datasets
    def __init__(self, images, labels, batch_size=256, shuff=True, multiclass=True, n_cl=5, angles_list=None):
        self.images, self.labels = images, labels
        self.batch_size = batch_size
        self.indices = np.arange(self.images.shape[0])
        self.batch_ids = np.arange(np.ceil(self.images.shape[0] / float(self.batch_size)), dtype=int)
        self.shuffle = shuff
        self.multiclass = multiclass
        self.num_class = n_cl
        #self.aug_train = ImageDataGenerator(rotation_range=90, fill_mode='reflect', dtype=float)
        self.angles = angles_list

    def __len__(self):
        return int(np.ceil(self.images.shape[0] / float(self.batch_size)))
    
    def on_epoch_end(self):
        # to shuffle indices each epoch
        if self.shuffle:
            np.random.shuffle(self.batch_ids)

    def __getitem__(self, idx):
        idx = self.batch_ids[idx]
        load_ids = list(self.indices[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_x = self.images[load_ids]
        #batch_x = batch_x/256.
        batch_y = self.labels[load_ids]
        if self.multiclass: batch_y = to_categorical(batch_y, num_classes=self.num_class)
        #params = self.aug_train.get_random_transform(batch_x.shape[1:])
        rot_angles = np.random.choice(self.angles, batch_x.shape[0])
        for i in range(batch_x.shape[0]):
            #batch_x[i] = self.aug_train.apply_transform(batch_x[i], params)
            for i_pol in range(batch_x.shape[-1]):
                batch_x[i,...,i_pol] = skimage.transform.rotate(batch_x[i,...,i_pol], angle=rot_angles[i], mode='symmetric', order=3)
        return batch_x[...,np.newaxis], batch_y
    
    
    
    
### FOR CHECKPOINT PREDICTIONS ###
from keras.models import load_model
def pred_checkpoints(dset=None, mod_dir=None, batch=256, N_ep=100, s_='Carbon-60keV', b_='beta', tr_type='no_rot', verb=1, rpr=False, n_rpr=10, pred_mode='mean'):
    if rpr: tr_type += '_rpr'
    print('Predicting for checkpoints', s_, 'vs', b_)
    start = datetime.now()
    with h5py.File(dset+s_+'_'+b_+'.h5','r') as data_sb:
        X_val, y_val = data_sb['val/images'][...], data_sb['val/labels'][...]
        X_val = X_val[...,np.newaxis]/256.
    #val_gen = Batch_data_generator(X_val,y_val,batch_size=batch, shuff=False)
    print('\tNumber of validation samples',X_val.shape[0])
    mod_dir += tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'
    #if os.path.exists('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'):
    #    for fname in os.listdir('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'):
    #        if pred_mode in fname
    #    shutil.rmtree('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/')
    for mod_name in os.listdir(mod_dir):
        fold = datetime.now()
        K.clear_session()
        if verb: print('\n','\t\tcheckpoint:',mod_name)
        conv4_3d_res = load_model(mod_dir+mod_name,custom_objects={'swish':swish})
    #preds = conv4_3d_res.predict_generator(val_gen, workers=5, max_queue_size=10, use_multiprocessing=True, verbose=verb)
        if rpr:
            preds=[]
            for idx in tqdm(range(X_val.shape[0]//batch + 1)):
                X_btch = np.copy(X_val[batch*idx:batch*(idx+1)])
                preds = np.append(preds,rot_predict(model=conv4_3d_res, batch_x=X_btch, n_rot=n_rpr, mode=pred_mode))
                gc.collect()
        else: preds = conv4_3d_res.predict(X_val, batch_size=batch, verbose=verb)
        if verb: print('\t\ttime ', datetime.now()-fold)
        if not rpr: pred_mode = ''
        #print('ROC AUC ',roc_auc_score(y[n]['/images/val'],preds))
        preds = np.vstack((np.squeeze(preds),y_val[...])).T
        if not os.path.exists('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'):
            os.makedirs('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/')
        if verb: print('\tSaving prediction to','outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'+pred_mode+'_'+mod_name[:-3]+'.txt')
        np.savetxt('outputs/conv4_v4/preds/checkpoints/'+tr_type+'/'+s_+'_'+b_+'/ep'+str(N_ep)+'/'+pred_mode+'_'+mod_name[:-3]+'.txt',preds)
        del conv4_3d_res
    print('\tvalidation time ', datetime.now()-start)
    del X_val, y_val
    _=gc.collect()