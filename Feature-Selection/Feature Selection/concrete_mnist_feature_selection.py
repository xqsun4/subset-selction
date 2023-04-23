# -*- coding: utf-8 -*-
"""
    A Unified Perspective on Regularization and Perturbation in Differentiable Subset Selection

    function: select k pixels from MNIST datasets by using Concrete Autoencoder[1].
    
    References:
                1.Balın et al. Concrete Autoencoders: Differentiable Feature Selection and Reconstruction
                Proceedings of the 36th International Conference on Machine Learning, PMLR 97:444-453, 2019.
                
"""

import math
from keras import backend as K
from keras import Model
from keras.layers import Layer, Softmax, Input
from keras.callbacks import EarlyStopping
from keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
from os.path import join
import pandas as pd
import time 

starttime = time.time()


def f(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(784)(x)
    return x

def g(x):
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(10)(x)
    x = Softmax()(x)
    return x

class ConcreteSelect(Layer):
    
    def __init__(self, output_dim, start_temp = 10.0, min_temp = 0.1, alpha = 0.9999, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        super(ConcreteSelect, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.temp = self.add_weight(name = 'temp', shape = [], initializer = Constant(self.start_temp), trainable = False) #temperature
        self.logits = self.add_weight(name = 'logits', shape = [self.output_dim, input_shape[1]], initializer = glorot_normal(), trainable = True)
        super(ConcreteSelect, self).build(input_shape)
        
    def call(self, X, training = None):
        #------
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0) #uniform from [1e-7, 1]
        gumbel = -K.log(-K.log(uniform))
        
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha)) #assign temp*alpha -> temp?
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits) #prob of each pixel 
        
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1]) #one_hot label
        
        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))
        
        return Y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    
class StopperCallback(EarlyStopping):
    
    def __init__(self, mean_max_target = 0.90):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max', baseline = self.mean_max_target)
    
    def on_epoch_begin(self, epoch, logs = None):
        print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature', K.get_value(self.model.get_layer('concrete_select').temp))
        print('-'*20, 'sample shape: ', K.get_value(self.model.get_layer('concrete_select').logits.shape), '-'*20)

    
    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        return monitor_value

    
if __name__ =="__main__":
    learning_rate = 0.001
    for i in range(10):
        k = 50
        a = ConcreteSelect(output_dim=k, start_temp = 3.0, min_temp = 0.1,  name = 'concrete_select')
        from normal_gumbel_autoencoder import ConcreteAutoencoderFeatureSelector
        from keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical
        from keras.layers import Dense, Dropout, LeakyReLU, Softmax
        import numpy as np
    
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (len(x_train), -1))
        x_test = np.reshape(x_test, (len(x_test), -1))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)
        
        inputs = Input(shape = x_train.shape[1:])
        selected_feature = a(inputs)
        outputs = f(selected_feature)
        
        model = Model(inputs, outputs)
        print(model.summary())
    
        model.compile(Adam(learning_rate), loss = 'mean_squared_error')
        
        batch_size = max(len(x_train) // 256, 16)
        print(batch_size)
        
        stopper_callback = StopperCallback()
        
        # print(model.summary())
        model.fit(x_train, x_train, batch_size = 234, epochs = 800, callbacks = [stopper_callback])
        
        concrete_layer = model.get_layer(name = 'concrete_select')
        probabilities = K.get_value(K.softmax(concrete_layer.get_weights()[0]))
        indices = K.get_value(K.argmax(concrete_layer.get_weights()[0]))
        print('prob: ', probabilities)
        print('indices:', indices)
        
        expand = tf.one_hot(indices, 784)
        topk = tf.reduce_sum(expand, axis = 0).numpy()
        print(np.sum(topk))
        print('topk:', topk)
        
        column_values = ['0']
        df = pd.DataFrame(data =topk, columns = column_values )
        path = 'table'
        text = 'concrete_mnist_feature_selection_{}_{}.csv'.format(i, k)
        df.to_csv(join(path, text), index=False, sep=',')
    
    endtime = time.time()
    print('time: ', endtime - starttime)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    