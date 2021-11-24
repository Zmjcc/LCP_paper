import numpy as np

np.random.seed(2021)
from scipy import io
from sklearn import preprocessing
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

from option import parge_config
from data_preprocess import data_process
net_name = 'init_net'
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
    epochs = 20
test_length = 2000
dataset_root = '/home/zmj/LCP_dataset/dataset/'
data_root = dataset_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,B, SNR_dB)
train_mode = 'train'

# dataset,test_dataset,H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su,dataset_bar,test_dataset_bar = \
#                                         data_process(data_root,Nt,Nr,dk,K,B,SNR_dB,SNR_channel_dB,test_length,data_mode)

prefix = 'learn_from_H_bar'
##################
# %% supervised training
def backbone(data,Nt,Nr,dk,K,B):
    def vector_norm_l1(vec):
        v_norm,_ = tf.linalg.normalize(vec,ord=1,axis=1)
        return v_norm
    def vector_norm_l2(vec):
        v_norm, _ = tf.linalg.normalize(vec, ord=2, axis=1)
        return v_norm

    data = Reshape((K*dk*B,K*dk*B,1))(data)

    if K*dk>12:
        x = Conv2D(filters=16, kernel_size=(7, 7))(data)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    else:
        x = data
    x = Conv2D(filters=8, kernel_size=(5, 5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    for i in range(3):
        x = Conv2D(filters=4, kernel_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
    x = Conv2D(filters=2, kernel_size=(3, 3))(x)
    x = Flatten()(x)
    x_tp = BatchNormalization()(x)
    tp_list = Dense(K * dk)(x_tp)
    tp_list = BatchNormalization()(tp_list)
    tp_list = Activation('relu')(tp_list)
    tp_list = Dense(K * dk )(tp_list)
    tp_list = Activation('sigmoid')(tp_list)
    tp_list = Lambda(vector_norm_l1)(tp_list)
    x_up = BatchNormalization()(x)
    up_list = Dense(K * dk * B)(x_up)
    up_list = BatchNormalization()(up_list)
    up_list = Activation('relu')(up_list)
    up_list = Dense(K * dk *B)(up_list)
    up_list = Activation('sigmoid')(up_list)
    up_list = Lambda(vector_norm_l1)(up_list)
    x_q = BatchNormalization()(x)
    q_list = Dense(K * dk * B * 2)(x_q)
    q_list = BatchNormalization()(q_list)
    q_list = Activation('relu')(q_list)
    q_list = Dense(K * dk * B * 2)(q_list)
    q_list = Activation('tanh')(q_list)
    prediction = Concatenate(axis=1)([tp_list, up_list,q_list])
    return prediction

def weighted_mse_loss(y_true, y_pred):
    pa_list_pred = y_pred[:, :K * dk]
    lambda_list_pred = y_pred[:, K * dk:(K*dk + K*dk*B)]
    q_list_pred = y_pred[:,(K*dk + K*dk*B):]

    pa_list_true = y_true[:, :K * dk]
    lambda_list_true = y_true[:, K * dk: (K*dk + K*dk*B)]
    q_list_true = y_true[:,(K*dk + K*dk*B):]

    mse_pa = tf.reduce_mean(tf.square((pa_list_true - pa_list_pred)), axis=-1)
    mse_lambda = tf.reduce_mean(tf.square((lambda_list_true - lambda_list_pred)), axis=-1)
    mse_q = tf.reduce_mean(tf.square((q_list_true - q_list_pred)), axis=-1)
    loss = mse_pa + mse_lambda + mse_q
    return loss


def su_net(Nt,Nr, K, B, dk, lr):
    data = Input(shape=(K*dk*dk*K*B*B))
    prediction = backbone(data,Nt,Nr,dk,K,B)
    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=weighted_mse_loss, optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-2
K_list = [8,9,10,11,12,13,14,15,16]
#K_list = [10]
dk_list = [2]
for dk in dk_list:
    for K in K_list:
        model = su_net(Nt,Nr, K, B, dk, lr)
        su_model_path = dataset_root + 'test_model/DUU_MISO_models_%d_%d_%d_%d_%d_%d_%s_2_RB.h5' % (Nt, Nr, K, dk, SNR_dB,SNR_channel_dB,net_name)
        model.save_weights(su_model_path)
# checkpointer = ModelCheckpoint(su_model_path, verbose=1, save_best_only=True, save_weights_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-5, min_lr=1e-5)
#
# if train_mode == 'train':
#     model.fit(dataset_bar, labelset_su, epochs=epochs, batch_size=batch_size, verbose=2, \
#               validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])




# CUDA_VISIBLE_DEVICES=0 python train_main.py  --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1
