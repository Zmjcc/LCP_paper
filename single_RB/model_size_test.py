import numpy as np

np.random.seed(2021)
from scipy import io
from sklearn import preprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Flatten, Permute, Reshape, Input, \
    BatchNormalization, Concatenate, Add, Lambda, GlobalAveragePooling1D, Concatenate, GlobalAvgPool1D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as BK

import os
import hdf5storage
tf.config.run_functions_eagerly(True)

from option import parge_config
from data_preprocess import data_process

args = parge_config()
Nt = args.Nt
Nr = args.Nr
K = args.K
dk = args.dk
SNR_dB = args.SNR
B = args.B
SNR = 10 ** (SNR_dB / 10)
p = 1
sigma_2 = 1 / SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode
batch_size = args.batch_size
epochs = args.epoch
if data_mode=='debug':
    epochs = 1
test_length = 2000
dataset_root = '/home/zmj/LCP_dataset/dataset/'
data_root = dataset_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,B, SNR_dB)
train_mode = 'train'

# dataset,test_dataset,dataset_bar,test_dataset_bar,H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su = \
#                                         data_process(data_root,Nt,Nr,dk,K,SNR_dB,SNR_channel_dB,test_length,data_mode)
prefix = 'learn_from_H_bar'
net_name = 'init'
##################
# %% supervised training
def backbone(data,Nt,Nr,dk,K):
    def vector_norm(vec):
        #vector shape:B*L

        v_norm,_ = tf.linalg.normalize(vec,ord=1,axis=1)
        return v_norm
    net = 'sigmoid'
    if net=='softmax':
        data = Reshape((Nt,2*K*dk))(data)
        x = Conv1D(filters=int(40),kernel_size=7)(data)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv1D(filters=int(20), kernel_size=5)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv1D(filters=int(10), kernel_size=3)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        tp_list = Dense(K * dk)(x)
        tp_list = Activation('softmax')(tp_list)
        up_list = Dense(K * dk)(x)
        up_list = Activation('softmax')(up_list)
        prediction = Concatenate(axis=1)([tp_list, up_list])
    else:
        data = Reshape((K*dk,K*dk,1))(data)
        #import ipdb;ipdb.set_trace()
        if K*dk>12:
            x = Conv2D(filters=16, kernel_size=(7, 7))(data)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        else:
            x = data
        x = Conv2D(filters=8, kernel_size=(5, 5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=4, kernel_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        tp_list = Dense(K * dk)(x)
        tp_list = Activation('sigmoid')(tp_list)
        tp_list = Lambda(vector_norm)(tp_list)
        up_list = Dense(K * dk)(x)
        up_list = Activation('sigmoid')(up_list)
        up_list = Lambda(vector_norm)(up_list)
        prediction = Concatenate(axis=1)([tp_list, up_list])
    return prediction

def weighted_mse_loss(y_true, y_pred):
    p_list_pred = y_pred[:, :K * dk]
    q_list_pred = y_pred[:, K * dk:2 * K * dk]
    p_list_true = y_true[:, :K * dk]
    q_list_true = y_true[:, K * dk:2 * K * dk]
    mse_p = tf.reduce_mean(tf.square((p_list_true - p_list_pred)), axis=-1)
    mse_q = tf.reduce_mean(tf.square((q_list_true - q_list_pred)), axis=-1)
    loss = mse_p + mse_q
    return loss


def su_net(Nt,Nr, K, dk, lr):
    data = Input(shape=(K*dk*K*dk))
    prediction = backbone(data,Nt,Nr,dk,K)
    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=weighted_mse_loss, optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-2

K_list = [8,9,10,11,12,13,14,15,16]
dk_list = [2]
for dk in dk_list:
    for K in K_list:
        model = su_net(Nt,Nr, K,  dk, lr)
        su_model_path = dataset_root + 'test_model/DUU_MISO_models_%d_%d_%d_%d_%d_%d_%s_single_RB.h5' % (Nt, Nr, K, dk, SNR_dB,SNR_channel_dB,net_name)
        model.save_weights(su_model_path)

# CUDA_VISIBLE_DEVICES=0 python learn_from_bar.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2
