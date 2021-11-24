import numpy as np
import os
np.random.seed(2020)
import tensorflow as tf

from scipy import io
# import os
import os
import hdf5storage
import random
from sklearn import preprocessing
import logging

#tf.config.run_functions_eagerly(True)



# %% load and construct data
from option import parge_config

args = parge_config()
Nt = args.Nt
Nr = args.Nr
K = args.K
B = args.B
dk = args.dk
SNR_dB = args.SNR
SNR = 10 ** (SNR_dB / 10)
p = 1
sigma_2 = 1 / SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode

mode = 'train'
data_mode = 'debug'

dataset_root = '/home/zmj/LCP_dataset/dataset/'
data_root = dataset_root + 'channel_dataset.mat'
if data_mode =='debug':
    data_root = dataset_root + 'test_channel_dataset.mat'
else:
    data_root = dataset_root + 'channel_dataset.mat'
channels = hdf5storage.loadmat(data_root)['H_list'][:, :, :, :K, :B]
channels = np.transpose(channels,(0,1,2,4,3))
model_root = './model/'
factor = 1


def minus_sum_rate_loss(y_true, y_pred,Nt,Nr,dk,K,B,p,sigma_2):
    '''
    y_true is the channels
    y_pred is the predicted beamformers
    notice that, y_true has to be the same shape as y_pred
    '''
    ## construct complex data  channel shape:Nt,Nr,2*K   y_pred shape:Nt,dk,K,2
    y_true = tf.cast(tf.reshape(y_true, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = y_true[:, :, :, :, :K] + 1j * y_true[:, :, :, :, K:]
    y_pred = tf.cast(y_pred, tf.complex128)
    V0 = y_pred[:, :, :, :, 0] + 1j * y_pred[:, :, :, :, 1]

    ## power normalization of the predicted beamformers
    # VV = tf.matmul(V0,tf.transpose(V0,perm=[0,2,1],conjugate = True))
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))

    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    sum_rate = 0.0
    # import ipdb;ipdb.set_trace()
    for k in range(K):
        for rb in range(B):
            H_k = tf.transpose(H[:, :, :, rb, k], perm=[0, 2, 1])  # NrxNt
            V_k = V[:, :, :, k]  # Ntx1
            signal_k = tf.matmul(H_k, V_k)
            signal_k_energy = tf.matmul(signal_k, tf.transpose(signal_k, perm=[0, 2, 1], conjugate=True))
            interference_k_energy = 0.0
            for j in range(K):
                if j != k:
                    V_j = V[:, :, :, j]
                    interference_j = tf.matmul(H_k, V_j)
                    interference_k_energy = interference_k_energy + tf.matmul(interference_j,
                                                                              tf.transpose(interference_j, perm=[0, 2, 1],
                                                                                           conjugate=True))
            SINR_k = tf.matmul(signal_k_energy,
                               tf.linalg.inv(interference_k_energy + sigma_2 * tf.eye(Nr, dtype=tf.complex128)))
            rate_k = tf.math.log(tf.linalg.det(tf.eye(Nr, dtype=tf.complex128) + SINR_k)) / tf.cast(tf.math.log(2.0),
                                                                                                    dtype=tf.complex128)
            sum_rate = sum_rate + rate_k
    sum_rate = tf.cast(tf.math.real(sum_rate), tf.float32)
    # loss
    loss = sum_rate
    return loss
def calculate_sinr(channel,Nt,Nr,dk,K,p,sigma_2,V):
    # channel shape:complex (B,Nt,Nr,2*K)
    # V shape:(B,Nt,dk,K,2)
    '''note here the Nr should be 1'''
    channel = tf.cast(channel,tf.complex128)
    channel = channel[:,:,:,:K] + 1j*channel[:,:,:,K:]
    V = tf.cast(V,tf.complex128)
    V = V[:,:,:,:,0] + 1j*V[:,:,:,:,1]
    sinr_list = list()
    for k in range(K):
        H_k = tf.transpose(channel[:, :, :, k], perm=[0, 2, 1])  # NrxNt
        V_k = V[:, :, :, k]  # Ntx1
        signal_k = tf.matmul(H_k, V_k)
        signal_k_energy = tf.matmul(signal_k, tf.transpose(signal_k, perm=[0, 2, 1], conjugate=True))
        interference_k_energy = 0.0
        for j in range(K):
            if j != k:
                V_j = V[:, :, :, j]
                interference_j = tf.matmul(H_k, V_j)
                interference_k_energy = interference_k_energy + tf.matmul(interference_j,
                                                                          tf.transpose(interference_j, perm=[0, 2, 1],
                                                                                       conjugate=True))
        SINR_k = tf.matmul(signal_k_energy,
                           tf.linalg.inv(interference_k_energy + sigma_2 * tf.eye(Nr, dtype=tf.complex128))) +1e-16

        sinr_list.append(SINR_k)
    sinr_list = tf.stack(sinr_list,3)
    sinr_list = tf.cast(tf.reshape(sinr_list,[-1,K]),tf.float64)
    return sinr_list



def EZF(channel,Nt,Nr,dk,K,B,p,sigma_2,P_return=False):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B,2 * K]), tf.complex128)
    H = H[:, :, :, :,:K] + 1j * H[:, :, :,:, K:]
    H = tf.transpose(H, [0, 2, 1, 3, 4])
    P = list()
    for user in range(K):
        H_this_user = tf.matmul(tf.transpose(H[:, :, :, 0, user], [0, 2, 1], conjugate=True), H[:, :, :, 0, user])
        for rb in range(1,B):
            H_this_user = H_this_user + tf.matmul(tf.transpose(H[:, :, :, rb, user], [0, 2, 1], conjugate=True), H[:, :, :, rb, user])
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :dk])
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    V = tf.matmul(P, tf.linalg.inv(tf.matmul(tf.transpose(P, [0, 2, 1], conjugate=True), P)))  # B*Nt*Kdk
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    # import ipdb;ipdb.set_trace()
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    if P_return:
        P = tf.reshape(P, [-1, Nt, 1, K * dk])
        P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=3), dtype=tf.float32)
        return V,P
    else:
        return V
def MIMO2MISO(channel,Nt,Nr,dk,K,B):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3,4])
    P = list()
    for user in range(K):
        for rb in range(B):
            H_this_user = H[:, :, :, rb, user]
            s, _, v = tf.linalg.svd(H_this_user)
            v = tf.matmul(v,tf.cast(tf.linalg.diag(s),tf.complex128))
            P.append((tf.math.conj(v[:, :, :dk])))
    P = tf.stack(P, axis=1)
    P = tf.reshape(tf.transpose(tf.reshape(P, [-1, K,B,Nt, dk]),[0,3,2,4,1]),[-1,Nt,1,B,K*dk])
    P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=4), dtype=tf.float32)
    return P

def MMSE_DUAL_PA(channel,Nt,Nr,dk,K,B,p,sigma_2,V_wmmse,U_wmmse,W_wmmse):
    #sinr_list shape: (B,K*dk)
    #channel shape:complex (B,Nt,1,K)
    lambda_list = list()
    q_list = list()
    pa_list = tf.norm(tf.reshape(tf.transpose(V_wmmse, [0, 1, 2, 4, 3]), [-1, Nt *dk* 2, K]), axis=1)**2
    for user in range(K):
        for rb in range(B):
            q_temp = tf.matmul(U_wmmse[:, :, :, rb, user], W_wmmse[:, :, :, rb, user])
            lambda_temp = tf.matmul(q_temp,U_wmmse[:,:,:,rb,user],adjoint_b = True)
            lambda_list.append(lambda_temp)
            q_list.append(q_temp)

    lambda_list = tf.reshape(tf.stack(lambda_list,1),[-1,K,B])
    lambda_list = p * lambda_list / tf.tile(tf.cast(tf.reshape(tf.norm(tf.reshape(lambda_list,[-1,K*B]),ord=1,axis=1),[-1,1,1]),tf.complex128),(1,K,B))
    q_list = tf.reshape(tf.stack(q_list,1),[-1,K,B])
    q_list = p * q_list / tf.tile(tf.cast(tf.reshape(tf.norm(tf.reshape(q_list,[-1,K*B]),ord=2,axis=1),[-1,1,1]),tf.complex128),(1,K,B))
    lambda_list = tf.cast(lambda_list,dtype=tf.float32)
    pa_list = tf.cast(pa_list,dtype=tf.float32)
    q_list = tf.reshape(q_list,[-1,K,B,1])
    q_list = tf.cast(tf.concat([tf.math.real(q_list),tf.math.imag(q_list)],axis=3),dtype=tf.float32)

    return lambda_list,pa_list,q_list

def pq2V(channel,Nt,Nr,dk,K,B,p,sigma_2,lambda_list,pa_list,q_list):
    # recover
    channel = tf.cast(channel,tf.complex128)
    channel = channel[:,:,:,:,:K] + 1j*channel[:,:,:,:,K:]  #new channel shape:(B,Nt,K)
    channel = tf.transpose(channel,[0,2,1,3,4])


    lambda_list = tf.cast(lambda_list,dtype=tf.complex128)


    pa_list = tf.cast(pa_list,dtype=tf.complex128)
    q_list = tf.cast(q_list,dtype=tf.complex128)
    q_list = q_list[:,:,:,0] + 1j*q_list[:,:,:,1]
    temp_inverse = sigma_2 * tf.eye(Nt, dtype=tf.complex128)
    for user in range(K):
        for rb in range(B):
            temp_inverse = temp_inverse + tf.tile(tf.reshape(lambda_list[:,user,rb],[-1,1,1]),[1,Nt,Nt])*tf.matmul(channel[:,:,:,rb,user],channel[:,:,:,rb,user],adjoint_a=True)
    temp_inverse = tf.linalg.inv(temp_inverse)

    V_norm = list()
    for user in range(K):
        HUW_this_user = tf.zeros([Nt,dk],dtype=tf.complex128)
        for rb in range(B):
            HUW_this_user = HUW_this_user + tf.tile(tf.reshape(q_list[:,user,rb],[-1,1,1]),[1,Nt,Nr])*tf.transpose(channel[:,:,:,rb,user],[0,2,1],conjugate=True)
        V_temp = tf.matmul(temp_inverse,HUW_this_user)
        V_temp, _ = tf.linalg.normalize(V_temp[:, :, 0] + 1e-16, axis=1)
        V_norm.append(V_temp)
    V_norm = tf.stack(V_norm,2)
    V = list()
    for k in range(K):
        V_temp = tf.tile(tf.reshape(tf.sqrt(pa_list[:,k]),(-1,1)),(1,Nt)) * V_norm[:,:,k]
        V.append(V_temp)
    V = tf.stack(V,2) #(B,Nt,K)
    V = tf.reshape(V,(-1,Nt,1,K,1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def WMMSE(channel,Nt,Nr,dk,K,B,p,sigma_2,UW_return):
    def update_WMMSE_U(H,V):
        U = list()
        trace_VV = 0
        for user in range(K):
            trace_VV = trace_VV + tf.linalg.trace(
                tf.matmul(V[:, :, :, user], V[:, :, :, user],adjoint_b=True))
        for rb in range(B):
            for user in range(K):
                HVVH = tf.zeros([Nr, Nr], dtype=tf.complex128)
                for k in range(K):
                    HV = tf.matmul(H[:, :, :, rb, user], V[:, :, :, k])
                    HVVH = HVVH + tf.matmul(HV, HV,adjoint_b=True)
                inverse_temp = tf.linalg.inv(sigma_2 / p * tf.tile(tf.reshape(trace_VV, (-1, 1, 1)), [1, Nr, Nr]) * tf.eye(Nr,dtype=tf.complex128) + HVVH)
                U_this_user = tf.matmul(tf.matmul(inverse_temp,H[:, :, :, rb,user]), V[:, :, :, user])
                U.append(U_this_user)
        U = tf.stack(U, 1)  # B*Nr*dk*K
        U = tf.transpose(tf.reshape(U,(-1,B,K,Nr,dk)),[0,3,4,1,2])
        # final U shape: B x Nr x dk x B x K
        return U
    def update_WMMSE_W(H,U,V):
        W = list()
        for user in range(K):
            for rb in range(B):
                HV = tf.matmul(H[:, :, :, rb, user], V[:, :, :, user])
                W_this_user = tf.linalg.inv(tf.eye(dk, dtype=tf.complex128) - tf.matmul(
                    U[:, :, :, rb, user], HV,adjoint_a=True))
                W.append(W_this_user)
        W = tf.stack(W, 1)
        W = tf.transpose(tf.reshape(W,(-1,K,B,dk,dk)),(0,3,4,2,1))
        return W
    def update_WMMSE_V(H,U,W):
        temp_B = tf.zeros([Nt, Nt], dtype=tf.complex128)
        for user in range(K):
            for rb in range(B):
                HHU = tf.matmul(H[:, :, :, rb, user], U[:, :, :, rb, user],adjoint_a=True)  # b*Nt*dk
                trace_UWU = sigma_2 / p * tf.linalg.trace(tf.matmul(tf.matmul(U[:, :, :, rb, user], W[:, :, :, rb, user]),
                                                                    U[:, :, :, rb, user],adjoint_b=True))
                temp_B = temp_B + tf.tile(tf.reshape(trace_UWU, (-1, 1, 1)), [1, Nt, Nt]) * tf.eye(Nt,
                                                                                                   dtype=tf.complex128) + tf.matmul(
                    tf.matmul(HHU, W[:, :, :, rb, user]), HHU,adjoint_b=True)

        temp_B_inverse = tf.linalg.inv(temp_B)

        V0 = list()
        VV = tf.zeros([batch_size, Nt, Nt], dtype=tf.complex128)

        for user in range(K):
            HUW = tf.matmul(tf.matmul(H[:, :, :, 0, user], U[:, :, :, 0, user],adjoint_a=True),W[:, :, :, 0,user])
            for rb in range(1,B):
                HUW = HUW + tf.matmul(tf.matmul(H[:, :, :, rb, user], U[:, :, :, rb, user],adjoint_a=True),W[:, :, :, rb,user])  # b*Nt*dk
            V0_this_user = tf.matmul(temp_B_inverse, HUW)
            V0.append(V0_this_user)
        V0 = tf.stack(V0, 3)
        V_norm,_ = tf.linalg.normalize(tf.reshape(V0,(-1,Nt*dk*K)),axis=1)
        V = tf.reshape(V_norm,(-1,Nt,dk,K))
        return V
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]  # B*Nt*Nr*K
    H = tf.transpose(H,(0,2,1,3,4))
    V = EZF(channel, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B, p=p, sigma_2=sigma_2)
    V = tf.cast(V, tf.complex128)
    V = V[:, :, :, :, 0] + 1j * V[:, :, :, :, 1]  # B*Nt*dk*K
    for i in range(50):
        U = update_WMMSE_U(H,V)
        W = update_WMMSE_W(H,U,V)
        V = update_WMMSE_V(H,U,W)
    V0 = V
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    V = tf.reshape(V,(-1,Nt,dk,K,1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    if UW_return:
        return U,W,V
    else:
        return V
def uw2v(channel,Nt,Nr,dk,K,B,p,sigma_2,u_flatten,w_flatten):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]  # B*Nt*Nr*K
    H = tf.transpose(H,(0,2,1,3,4))
    U = tf.cast(tf.reshape(u_flatten, [-1, Nr, dk, B, 2 * K]), tf.complex128)
    U = U[:, :, :, :, :K] + 1j * U[:, :, :, :, K:]
    W = tf.cast(tf.reshape(w_flatten, [-1, dk, dk, B, 2 * K]), tf.complex128)
    W = W[:, :, :, :, :K] + 1j * W[:, :, :, :, K:]
    temp_B = tf.zeros([Nt, Nt], dtype=tf.complex128)
    for user in range(K):
        for rb in range(B):
            HHU = tf.matmul(H[:, :, :, rb, user], U[:, :, :, rb, user],adjoint_a=True)  # b*Nt*dk
            trace_UWU = sigma_2 / p * tf.linalg.trace(tf.matmul(tf.matmul(U[:, :, :, rb, user], W[:, :, :, rb, user]),
                                                                U[:, :, :, rb, user],adjoint_b=True))
            temp_B = temp_B + tf.tile(tf.reshape(trace_UWU, (-1, 1, 1)), [1, Nt, Nt]) * tf.eye(Nt,
                                                                                            dtype=tf.complex128) + tf.matmul(
                tf.matmul(HHU, W[:, :, :, rb, user]), HHU,adjoint_b=True)
    temp_B_inverse = tf.linalg.inv(temp_B)
    V0 = list()
    VV = tf.zeros([batch_size, Nt, Nt], dtype=tf.complex128)
    for user in range(K):
        HUW = tf.matmul(tf.matmul(H[:, :, :, 0, user], U[:, :, :, 0, user],adjoint_a=True),W[:, :, :, 0,user])
        for rb in range(1,B):
            HUW = HUW + tf.matmul(tf.matmul(H[:, :, :, rb, user], U[:, :, :, rb, user],adjoint_a=True),W[:, :, :, rb,user])  # b*Nt*dk
        V0_this_user = tf.matmul(temp_B_inverse, HUW)
        V0.append(V0_this_user)
    V0 = tf.stack(V0, 3)
    V_norm,_ = tf.linalg.normalize(tf.reshape(V0,(-1,Nt*dk*K)),axis=1)
    V0 = tf.reshape(V_norm,(-1,Nt,dk,K))
    #V0 = V
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    V = tf.reshape(V,(-1,Nt,dk,K,1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V
data_num = len(channels)
#dataset = dataset[:data_num]
channels = channels[:data_num]
labelset = np.zeros((data_num, 2*dk*K))

u_flatten_dataset = np.zeros((data_num, Nr * dk * K * B * 2))
w_flatten_dataset = np.zeros((data_num, dk * dk * K * B * 2))

init_rate_list = []
final_rate_list = []

batch_size = 1000
total_iter = len(channels) // 1000
# import ipdb;ipdb.set_trace()
EZF_performance = []
WMMSE_performance = []

for i in range(total_iter):
    print('iteration:' + str(i))
    channel_iter = channels[i * 1000:(i + 1) * 1000, :]
    channel_iter = np.concatenate([np.real(channel_iter), np.imag(channel_iter)], axis=-1)
    EZF_output = EZF(channel_iter,Nt=Nt,Nr=Nr,dk=dk,K=K,B=B,p=p,sigma_2=sigma_2)
    EZF_performance.append(np.mean(minus_sum_rate_loss(channel_iter, EZF_output,Nt,Nr,dk,K,B,p,sigma_2))/B)
    WMMSE_U_output,WMMSE_W_output,WMMSE_V_output = WMMSE(channel_iter,Nt=Nt,Nr = Nr,dk=dk,K=K,B=B,p=p,sigma_2=sigma_2,UW_return=True)
    WMMSE_performance.append(np.mean(minus_sum_rate_loss(channel_iter, WMMSE_V_output,Nt,Nr,dk,K,B,p,sigma_2))/B)
    u_flatten_iter = tf.reshape(tf.concat([tf.math.real(WMMSE_U_output), tf.math.imag(WMMSE_U_output)], axis=-1),
                            [-1, 2 * Nr * dk * K * B])
    w_flatten_iter = tf.reshape(tf.concat([tf.math.real(WMMSE_W_output), tf.math.imag(WMMSE_W_output)], axis=-1),
                            [-1, 2 * dk * dk * K * B])
    u_flatten_dataset[i * 1000:(i + 1) * 1000, :] = u_flatten_iter
    w_flatten_dataset[i * 1000:(i + 1) * 1000, :] = w_flatten_iter

data_save_root = dataset_root + 'data/LUW_dataset_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,B, SNR_dB)

hdf5storage.savemat(data_save_root,
            {'U':u_flatten_dataset,'W':w_flatten_dataset,'H': channels})


logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(dataset_root + 'data/LUW_dataset_%d_%d_%d_%d_%d_%d.log' % (Nt, Nr, K, dk,B, SNR_dB))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('EZF sum rate:%.5f' % np.mean(EZF_performance))
logger.info('WMMSE sum rate:%.5f' % np.mean(WMMSE_performance))