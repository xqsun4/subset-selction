# -*- coding: utf-8 -*-
"""
    A Unified Perspective on Regularization and Perturbation in Differentiable Subset Selection
    
    function: evaluate informativeness of pixels selected by Perturbed Autoencoder
    
"""


import matplotlib
matplotlib.use('Agg')
figure_dir = 'figures'

import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from PIL import Image

import random
import datetime
import os
import pandas as pd

from concrete_estimator import run_experiment


def regularized_column_subset_selector_general(train, test, K, model_dir = None):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    ttrain = train
    train = (x_train, y_train)
    val = (x_val, y_val)
    path = 'table'
    text = 'regularized_kfold_mnist_selection_{}.csv'.format(K)
    indices_data = pd.read_csv(join(path, text))
    zero_logits = np.array(indices_data['0']).reshape((1, 784))
    _, feature_index = np.nonzero(zero_logits)
    
    indices = feature_index
    
    return ttrain[0][:, indices], test[0][:, indices]


def perturbed_column_subset_selector_general(train, test, K, model_dir = None):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    ttrain = train
    train = (x_train, y_train)
    val = (x_val, y_val)
    path = 'table'
    text = 'perturbed_kfold_mnist_selection_{}.csv'.format(K)
    indices_data = pd.read_csv(join(path, text))
    zero_logits = np.array(indices_data['0']).reshape((1, 784))
    _, feature_index = np.nonzero(zero_logits)
    
    indices = feature_index
    
    return ttrain[0][:, indices], test[0][:, indices]




def concrete_column_subset_selector_general(train, test, K, model_dir = None):
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1)
    ttrain = train
    train = (x_train, y_train)
    val = (x_val, y_val)
    
    probabilities = run_experiment('', train, val, test, K, [K * 3 // 2], 300, max(train[0].shape[0] // 256, 16), 0.001, 0.1)
    indices = np.argmax(probabilities, axis = 1)
    
    column_values = ['0']
    df = pd.DataFrame(data =indices, columns = column_values )
    print(df)
    path = 'table'
    text = r"concrete_column_mnist_feature_selection_{}.csv".format(K)
    df.to_csv(join(path, text), index=False, sep=',')
    
    return ttrain[0][:, indices], test[0][:, indices]



def load_data(fashion = False, digit = None, one_hot = False, normalize = False):
    if fashion:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit and digit <= 9:
        train = test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)

        for y in range(10):
            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])

        x_train = train[digit]
        x_test = test[digit]
    
    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(np.float32)
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)

    if one_hot:
        y_train_t = np.zeros((y_train.shape[0], 10))
        y_train_t[np.arange(y_train.shape[0]), y_train] = 1
        y_train = y_train_t
        y_test_t = np.zeros((y_test.shape[0], 10))
        y_test_t[np.arange(y_test.shape[0]), y_test] = 1
        y_test = y_test_t
    
    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train):]
    
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)



def load_mnist():
    train, test = load_data(fashion = False, normalize = True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size = 0.6)
    return (x_train, y_train), (x_test, y_test)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression

from skfeature.utility import unsupervised_evaluation
from sklearn.neighbors import KNeighborsClassifier
def eval_subset(train, test):
    n_clusters = len(np.unique(train[2]))
    
    clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
    clf.fit(train[0], train[2])
    DTacc = float(clf.score(test[0], test[2]))
    
    clf = KNeighborsClassifier(n_neighbors = 1, algorithm = 'brute', n_jobs = 1)
    clf.fit(train[0], train[2])
    acc = float(clf.score(test[0], test[2]))
    
    LR = LinearRegression(n_jobs = -1)
    LR.fit(train[0], train[1])
    MSELR = float(((LR.predict(test[0]) - test[1]) ** 2).mean())
    
    MSE = float((((decoder((train[0], train[1]), (test[0], test[1])) - test[1]) ** 2).mean()))
    
    max_iters = 10
    cnmi, cacc = 0.0, 0.0
    print('train[0]:', train[0])
    print('train[0]:', train[0].shape)
    for iter in range(max_iters):
        nmi, acc = unsupervised_evaluation.evaluation(train[0], n_clusters = n_clusters, y = train[2])
        cnmi += nmi / max_iters
        cacc += acc / max_iters
    print('nmi = {:.3f}, acc = {:.3f}'.format(cnmi, cacc))
    print('acc = {:.3f}, DTacc = {:.3f}, MSELR = {:.3f}, MSE = {:.3f}'.format(acc, DTacc, MSELR, MSE))
    return MSELR, MSE, acc, DTacc, float(cnmi), float(cacc)


def next_batch(samples, labels, num):
    # Return a total of `num` random samples and labels.
    idx = np.random.choice(len(samples), num)

    return samples[idx], labels[idx]

def decoder(train, test, debug = True, epoch_num = 200, dropout = 0.1):
    
    x_train, x_val, y_train, y_val = train_test_split(train[0], train[1], test_size = 0.1) #10% data to do cross validation
    train = (x_train, y_train)
    val = (x_val, y_val)
    
    D = train[0].shape[1]
    O = train[1].shape[1]
    
    learning_rate = 0.001
    
    best_val_cost = 1e100
    best_tt = 0
    for size in [D * 4 // 9, D * 2 // 3, D, D * 3 // 2]:
        
        tf.compat.v1.reset_default_graph()

        training = tf.compat.v1.placeholder(tf.bool, ())
        X = tf.compat.v1.placeholder(tf.float32, (None, D))
        TY = tf.compat.v1.placeholder(tf.float32, (None, O))

        net = tf.compat.v1.layers.dense(X, size, tf.nn.leaky_relu)
        net = tf.compat.v1.layers.dropout(net, rate = dropout, training = training)
        Y = tf.compat.v1.layers.dense(net, O)

        loss = tf.compat.v1.losses.mean_squared_error(Y, TY)
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        batch_size = max(train[0].shape[0] // 256, 16)
        batch_per_epoch = train[0].shape[0] // batch_size

        session_config = tf.compat.v1.ConfigProto

        
        val_cost = 0
        tt = np.zeros((0, train[1].shape[1]), np.float32)
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for ep in range(epoch_num):
                cost = 0
                for batch_n in range(batch_per_epoch):
                    imgs, yimgs = next_batch(train[0], train[1], batch_size)
                    # print('feature size', imgs.shape)
                    # print('size', yimgs.shape)
                    _, c = sess.run([train_op, loss], feed_dict = {X: imgs, TY: yimgs, training: True})
                    # print('loss', c)
                    cost += c / batch_per_epoch
                if debug and (ep + 1) % 50 == 0:
                    print('Epoch #' + str(ep + 1) + ' loss: ' + str(cost))
            for i in range(0, len(val[0]), batch_size):
                imgs, yimgs = val[0][i: i + batch_size], val[1][i: i + batch_size]
                c = sess.run(loss, feed_dict = {X: imgs, TY: yimgs, training: False})
                val_cost += c * len(imgs) / len(val[0])
            for i in range(0, len(test[0]), batch_size):
                imgs, yimgs = test[0][i: i + batch_size], test[1][i: i + batch_size]
                t = sess.run(Y, feed_dict = {X: imgs, training: False})
                tt = np.concatenate((tt, t))
        
        print(best_val_cost)
        print(val_cost)
        if best_val_cost > val_cost:
            best_val_cost = val_cost
            best_tt = tt
    
    return best_tt

import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from os.path import join

def eval_on_dataset(name, train, test, feature_sizes, debug = False):
    n_clusters = len(np.unique(train[1]))

    algorithms = [concrete_column_subset_selector_general, perturbed_column_subset_selector_general, regularized_column_subset_selector_general]

    
    
    #selected_indices = []
    alg_metrics = {}
    for alg in algorithms:
        nmi_results = {}
        acc_results = {}
        mseLR_results = {}
        mse_results = {}
        class_results = {}
        class_results_DT = {}
        #all_indices = {}
        for k in feature_sizes:
            print('k = {}, algorithm = {}'.format(k, alg.__name__))
            #indices = alg(x_train, enc.transform(y_train.reshape((-1, 1))).toarray(), k)
            tx_train, tx_test = alg((train[0], train[0]), (test[0], test[0]), k)
            #all_indices[k] = indices
            mseLR, mse, acc, acc_DT, cnmi, cacc = eval_subset((tx_train, train[0], train[1]), (tx_test, test[0], test[1]))
            mseLR_results[k] = float(mseLR)
            mse_results[k] = float(mse)
            class_results[k] = float(acc)
            class_results_DT[k] = float(acc_DT)
            nmi_results[k] = float(cnmi)
            acc_results[k] = float(cacc)
        #selected_indices.append((alg.__name__, all_indices))
        metrics = {'NMI': nmi_results, 'ACC': acc_results, 'MSELR': mseLR_results, 'MSE': mse_results, 'CLASS': class_results, 'CLASSDT': class_results_DT}
        alg_metrics[alg.__name__] = metrics

    with open(join(figure_dir, name), 'w') as fw:
        json.dump(alg_metrics,fw)
    
    return alg_metrics


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    
    
    for i in range(10):
        
        K = 50

        train, test = load_mnist()
        # eval_on_dataset('%d_perturbed_debug_mnist_%d.json' % (i, K), train, test, [K], True)
        eval_on_dataset('%d_concrete_regularized_perturbed_mnist_%d.json' % (i, K), train, test, [K], True)

        


if __name__ == '__main__':
    main()