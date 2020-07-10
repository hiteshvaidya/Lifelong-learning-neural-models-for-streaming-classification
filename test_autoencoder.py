#!/usr/bin/env python
# coding: utf-8

'''
This code is for only the left network (autoencoder) in the architecture.

:author: Hitesh Vaidya
'''

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import pickle as pkl
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

class Network():
    '''
    Class for left network i.e. autoencoder
    '''

    def __init__(self, n_x, n_z1, n_z2):
        '''
        Constructor for the class
        :param n_x: no. of input nodes
        :param n_z1: no. of nodes in layer 1
        :param n_z2: no. of nodes in layer 2
        '''
        self.params = []
        self.m1 = tf.Variable(
            tf.random.normal([n_x, n_z1], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='M1')
        self.n1 = tf.Variable(tf.random.normal([1, n_z1], mean=0.0, stddev=0.1,
                                               dtype=tf.dtypes.float32, seed=0),
                              name='n1')
        self.m2 = tf.Variable(
            tf.random.normal([n_z1, n_z2], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='M2')
        self.n2 = tf.Variable(tf.random.normal([1, n_z2], mean=0.0, stddev=0.1,
                                               dtype=tf.dtypes.float32, seed=0),
                              name='n2')
        self.m21 = tf.Variable(
            tf.random.normal([n_z2, n_z1], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='M21')
        self.n21 = tf.Variable(tf.random.normal([1, n_z1], mean=0.0, stddev=0.1,
                                                dtype=tf.dtypes.float32,
                                                seed=0), name='n21')
        self.m10 = tf.Variable(
            tf.random.normal([n_z1, n_x], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='M10')
        self.n10 = tf.Variable(tf.random.normal([1, n_x], mean=0.0, stddev=0.1,
                                                dtype=tf.dtypes.float32,
                                                seed=0), name='n10')

        self.params.append(self.m1)
        self.params.append(self.n1)
        self.params.append(self.m2)
        self.params.append(self.n2)
        self.params.append(self.m21)
        self.params.append(self.n21)
        self.params.append(self.m10)
        self.params.append(self.n10)

    def forward(self, x):
        '''
        Forward pass of the network
        :param x: input images
        :return: reconstructed nodes, left network loss
        '''
        Z1 = tf.matmul(x, self.m1) + self.n1
        Z1 = tf.nn.sigmoid(Z1)
        Z2 = tf.matmul(Z1, self.m2) + self.n2
        Z2 = tf.nn.sigmoid(Z2)
        Z1_hat = tf.matmul(Z2, self.m21) + self.n21
        Z1_hat = tf.nn.sigmoid(Z1_hat)
        Z0_hat = tf.matmul(Z1, self.m10) + self.n10
        Z0_hat = tf.nn.sigmoid(Z0_hat)

        at_loss = custom_MSE(Z1, Z1_hat) + custom_MSE(x, Z0_hat)
        return Z0_hat, at_loss

    def backward(self, x):
        '''
        Backward pass
        :param x: input images
        :return: None
        '''
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.01)
        with tf.GradientTape() as tape:
            Z0_hat, at_loss = self.forward(x)
        grads = tape.gradient(at_loss, self.params)
        optimizer.apply_gradients(zip(grads, self.params),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())

#load data
trainX = pd.read_csv('mnist_clean/trainX.tsv', sep="\t", header=None,
                     index_col=False, dtype=np.float32).to_numpy()
trainY = pd.read_csv('mnist_clean/trainY.tsv', sep="\t", header=None,
                     index_col=False, dtype=np.float32).to_numpy()
testX = pd.read_csv('mnist_clean/testX.tsv', sep="\t", header=None,
                    index_col=False, dtype=np.float32).to_numpy()
testY = pd.read_csv('mnist_clean/testY.tsv', sep="\t", header=None,
                    index_col=False, dtype=np.float32).to_numpy()
validX = pd.read_csv('mnist_clean/validX.tsv', sep="\t", header=None,
                     index_col=False, dtype=np.float32).to_numpy()
validY = pd.read_csv('mnist_clean/validY.tsv', sep="\t", header=None,
                     index_col=False, dtype=np.float32).to_numpy()

# Declare the network
net = Network(784, 512, 512)
n_epochs = 4
batch_size = 50

print('Configuration:')
print('Layers:', str([784, 512, 512]))
print('epochs:', 4)
print('batch_size:', 50)

# Declare losses
train_losses = []
test_losses = []
valid_losses = []

def custom_bce(y, p, offset=1e-7):  # 1e-10
    '''
    calculates Binary Cross Entropy value
    :param y: true labels
    :param p: predicted output
    :param offset: threshold for decimal value
    :return: BCE loss value
    '''
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = -tf.reduce_mean(
        y * tf.math.log(p_) + (1.0 - y) * tf.math.log(1.0 - p_), axis=1)
    return tf.reduce_mean(vec_bce)

def custom_KLD(p, q, offset=1e-7):  # q is model predictive/approximating
    # distribution, p is target distribution
    '''
    Calculates KL-Divergence
    :param p: true labels
    :param q: predicted output
    :param offset: threshold for decimal value
    :return: KL-Divergence
    '''
    q_ = tf.clip_by_value(q, offset, 1 - offset)
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = tf.reduce_sum(p_ * (tf.math.log(p_) - tf.math.log(q_)), axis=-1)
    return tf.reduce_mean(vec_bce)

def custom_MSE(y_true, y_pred, offset=1e-7):
    '''
    MSE loss function
    :param y_true: expected correct label
    :param y_pred: predicted output
    :param offset: decimal threshold
    :return: MSE loss value
    '''
    y_true = tf.clip_by_value(y_true, offset, 1-offset)
    y_pred = tf.clip_by_value(y_pred, offset, 1-offset)
    vec = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=1)
    return tf.reduce_mean(vec, axis=0)

def divide_chunks(l, n):
    '''
    Divides a list into batches of given size
    :param l: list
    :param n: batch size
    :return: object of subsets of l having sizes n
    '''
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def batch_loader(data, batch_size):
    '''
    Create batches of indices of data
    :param data: dataset
    :param batch_size: batch size
    :return: batches of data
    '''
    indices = np.arange(len(data))
    for _ in range(5): np.random.shuffle(indices)
    batches = np.asarray(list(divide_chunks(indices, batch_size)))
    return batches

def train_metrics(train_record):
    '''
    Calculates train loss and accuracy
    :return: train losses
    '''
    loss_at = 0
    size = 0
    nums = np.random.choice(trainY.shape[0], int(trainY.shape[0] * 0.4))
    batches = batch_loader(trainY[nums], 32)
    for batch in batches:
        Z0_hat, batch_loss = net.forward(trainX[batch])
        size += batch.shape[0]
        loss_at += batch_loss * batch.shape[0]
    print('\nTrain loss =', loss_at/size)
    train_record.append(loss_at / size)
    return train_record

def valid_metrics(valid_record):
    '''
    Calculates validation loss and accuracy
    :return: Validation losses
    '''
    loss_at = 0
    size = 0
    nums = np.random.choice(validY.shape[0], 1000)
    batches = batch_loader(validY[nums], 20)
    for batch in batches:
        Z0_hat, batch_loss = net.forward(validX[batch])
        size += batch.shape[0]
        loss_at += batch_loss * batch.shape[0]
    print('Validation loss =', loss_at/size)
    valid_record.append(loss_at / size)
    return valid_record

def test_metrics(test_record):
    '''
    Calculate test loss
    :param test_record: test losses
    :return: test losses
    '''
    loss_at = 0
    size = 0
    batches = batch_loader(testY, 32)
    for batch in batches:
        Z0_hat, batch_loss = net.forward(testX[batch])
        size += batch.shape[0]
        loss_at += batch_loss * batch.shape[0]
    print('Test loss =', loss_at / size)
    test_record.append(loss_at / size)
    return test_record

# declare array for collecting train loss
# train_losses[t] = [n_epochs x n_tasks]

# train_accuracy = []
# valid_accuracy = []
# test_accuracy = []

# calculate initial loss values before training network
train_losses = train_metrics(train_losses)
valid_losses = valid_metrics(valid_losses)
test_losses = test_metrics(test_losses)

# train and calculate BWT, FWT
tqdm.write('epochs running')

for epoch in tqdm(range(n_epochs)):
    batches = batch_loader(trainX, batch_size)
    for batch in batches:
        net.backward(trainX[batch])

    # calculated train, valid, test losses
    train_losses = train_metrics(train_losses)
    valid_losses = valid_metrics(valid_losses)
    test_losses = test_metrics(test_losses)

data_dict = {'train_loss': train_losses,
             'valid_loss': valid_losses,
             'test_loss': test_losses,
             # 'train_acc': train_accuracy,
             # 'valid_acc': valid_accuracy,
             # 'test_acc': test_accuracy
             }

pkl.dump(data_dict, open('results/at_data_dict.pkl', 'wb'))


def plot_loss():
    '''
    plots task wise train and validation losses of the model
    :return: None
    '''
    plt.plot(train_losses, label='Train loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss values')
    plt.title('Autoencoder loss')
    plt.legend(loc='upper right')
    plt.show()


# def plot_accuracy():
#     '''
#     plots task wise train and validation accuracies of the model
#     :return:
#     '''
#     plt.plot(train_accuracy, label='Train accuracy')
#     plt.plot(valid_accuracy, label='Validation accuracy')
#     plt.xlabel('epochs')
#     plt.ylabel('accuracy values')
#     plt.title('Autoencoder accuracy')
#     plt.legend(loc='upper right')
#     plt.show()

def plot_fig():
    '''
    Display input image and regenerated image
    :return: None
    '''
    # nums = np.random.choice(testX.shape[0], 3)
    trainX_Z0_hat, at_loss = net.forward(tf.reshape(trainX[0], [1,784]))
    validX_Z0_hat, at_loss = net.forward(tf.reshape(validX[0], [1,784]))
    testX_Z0_hat, at_loss = net.forward(tf.reshape(testX[0], [1,784]))

    trainX_Z0_hat = tf.reshape(trainX_Z0_hat, [-1])
    validX_Z0_hat = tf.reshape(validX_Z0_hat, [-1])
    testX_Z0_hat = tf.reshape(testX_Z0_hat, [-1])

    results = [[tf.dtypes.cast(255 * trainX[0], tf.int32), tf.dtypes.cast(
                    255 * trainX_Z0_hat, tf.int32)],
               [tf.dtypes.cast(255 * validX[0], tf.int32), tf.dtypes.cast(
                    255 * validX_Z0_hat, tf.int32)],
               [tf.dtypes.cast(255 * testX[0], tf.int32), tf.dtypes.cast(
                    255 * testX_Z0_hat, tf.int32)]
               ]

    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax[0, 0].imshow(tf.reshape(results[0][0], [28, 28]).numpy())
    ax[0, 1].imshow(tf.reshape(results[0][1], [28, 28]).numpy())
    ax[1, 0].imshow(tf.reshape(results[1][0], [28, 28]).numpy())
    ax[1, 1].imshow(tf.reshape(results[1][1], [28, 28]).numpy())
    ax[2, 0].imshow(tf.reshape(results[2][0], [28, 28]).numpy())
    ax[2, 1].imshow(tf.reshape(results[2][1], [28, 28]).numpy())
    plt.show()

    fig, ax = plt.subplots(nrows=3, ncols=2)
    ax[0, 0].hist(results[0][0])
    ax[0, 1].hist(results[0][1])
    ax[1, 0].hist(results[1][0])
    ax[1, 1].hist(results[1][1])
    ax[2, 0].hist(results[2][0])
    ax[2, 1].hist(results[2][1])
    plt.show()

    def get_median(arr):
        a = tf.size(arr).numpy()
        if a % 2 != 0:
            median = arr[a // 2]
        else:
            median = (arr[a // 2] + arr[(a // 2) - 1]) / 2
        return median

    for row in range(3):
        mean1 = tf.reduce_mean(results[row][0])
        mean2 = tf.reduce_mean(results[row][1])
        print('Mean of input =', mean1.numpy(), end=',\t')
        print('Mean of output =', mean2.numpy())
        median1 = get_median(tf.sort(results[row][0], axis=-1,
                                     direction='ASCENDING'))
        median2 = get_median(tf.sort(results[row][1], axis=-1,
                                     direction='ASCENDING'))
        print('Median of input =', median1.numpy(), end=',\t')
        print('Median of output =', median2.numpy())
        print(
            '----------------------------------------------------------------')


plot_loss()
plot_fig()

print('------------------------------------------------------')
print('Test loss:', test_losses)
# print('Test accuracy:', test_accuracy)
print('------------------------------------------------------')
print('Experiment completed')