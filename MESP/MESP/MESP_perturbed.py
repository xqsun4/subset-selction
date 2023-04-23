# -*- coding: utf-8 -*-
"""
    A Unified Perspective on Regularization and Perturbation in Differentiable Subset Selection

    function: select subsets with cardinality k for MESP by perturbed relaxation
    
    perturbations: Berthet, Quentin, et al. "Learning with differentiable pertubed optimizers." 
                   Advances in neural information processing systems 33 (2020): 9508-9519.
"""
import numpy as np
import scipy.io as scio
import heapq
from itertools import combinations
import tensorflow as tf
import perturbations.perturbations as pp


def argtopk(x, axis=-1):
    k = 120
    size = tf.shape(x)
    updates = tf.constant(tf.ones(k*size[0]))
    dim = tf.reshape(tf.range(size[0]), (-1, 1)) 
    dim_slice = tf.tile(dim, [1, k])
    index = tf.math.top_k(x, k=k).indices
    indices = tf.stack([tf.reshape(dim_slice, (-1, 1)), tf.reshape(index, (-1, 1))], axis=1)
    indices = tf.reshape(indices, (-1, 2))
    output = tf.scatter_nd(indices, updates, size)
    return output



datafile = 'D://xiangqian_exp2//MESP//data2000' 
n = 2000
k = 120
'''change the num'''
data = scio.loadmat(datafile)
C = np.array(data['C'])
trun_C = C*0.05 # trun_C = C*0.01 for n = 90, trun_C = C*0.1 for n=124, trun_C = C*0.05 for n = 2000

trun_C = tf.cast(tf.constant(trun_C), dtype='float32')
size = tf.shape(trun_C)

theta = tf.Variable(tf.random.normal([size[0]]), trainable=True, name = 'score')

def MESP_linx_loss(theta, C):
    size = tf.shape(C)
    pert_argtopk = pp.perturbed(argtopk,
                                          num_samples=10000,
                                          sigma=0.5,
                                          noise='gumbel',
                                          batched=False)
    S = pert_argtopk(theta)
    S = tf.cast(tf.linalg.diag(S), dtype=float)
    I = tf.eye(size[0])
    tensor1 = tf.matmul(C, S)
    eq = tf.matmul(tensor1, C)
    loss = -tf.math.log(tf.linalg.det(eq +I -S))
    return loss



optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

trainable_var = [theta]

    
epoch = 100



epoch_plot = list()
loss_plot = list()
    
for i in range(epoch):
    with tf.GradientTape() as tape:
        loss =  MESP_linx_loss(theta, trun_C)
    gradients = tape.gradient(loss, trainable_var)
    optimizer.apply_gradients(zip(gradients, trainable_var))
    if i%10==0:
        print('-'*10)
        print('Epoch: ', i, '; Loss: ', loss.numpy())
    
    epoch_plot.append(i+1)
    loss_plot.append(loss)
    

theta = theta.numpy()
print(theta)
S = heapq.nlargest(k, range(len(theta)), theta.__getitem__)
det = np.linalg.det(C[np.ix_(S, S)])
CS = np.log(np.linalg.det(C[np.ix_(S, S)])) # log determinant
# # print(theta)
print('-'*10, 'perturbed optimizer method', '-'*10)
print('C dimension is:', n, '; subset dimension is:', k)
print('approximate subset:', S)
pert_argtopk = pp.perturbed(argtopk,
                                      num_samples=1000,
                                      sigma=0.5,
                                      noise='gumbel',
                                      batched=False)
y = pert_argtopk(theta).numpy()
print(y)
print('appro subset prob', y[S])
# # print(heapq.nlargest(k, theta))
print('determinant:', det)
print('log maxmimum entropy:', CS)



    
    

    
    
    
    
    
    
    
    
    
    
    
    
