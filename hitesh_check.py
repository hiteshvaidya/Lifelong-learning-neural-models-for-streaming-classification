#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle as pkl

# In[15]:





# In[16]:


def relabel(labels):
    '''
    task wise relabel the dataset as combination of 0's and 1's
    '''
    new_labels = np.empty((0, labels.shape[0]), dtype=np.float32)
    for task in range(10):
        positives = np.where(labels[:, task] == 1)[0]
        task_labels = np.zeros(labels.shape[0], dtype=np.float32)
        task_labels[positives] = 1.0
        new_labels = np.vstack((new_labels, task_labels))
    #     new_labels = tf.convert_to_tensor(new_labels, dtype=tf.float32)
    return new_labels


def batch_loader(labels, batch_size, class_bal=False):
    '''
    load random batches of data
    '''
    # if class balance is not required in every batch
    if not class_bal:
        indices = np.arange(labels.shape[0])
        for _ in range(5): np.random.shuffle(indices)
        batches = np.asarray(list(divide_chunks(indices, batch_size)))
        return batches

    # if class balance is needed in every batch
    else:
        positives = np.where(labels == 1)[0]
        negatives = np.arange(labels.shape[0])
        negatives = np.delete(negatives, positives)
        np.random.shuffle(negatives)
        np.random.shuffle(positives)
        task_batch = []
        # create batches by iteratively scraping out chunks out of positives array
        while positives.shape[0] > 0:
            if len(positives) >= batch_size / 2:
                # create a batch such that positive (batch_size/2) is added with sampled negatives (batch_size/2)
                temp = np.concatenate((positives[:batch_size // 2], np.random.choice(negatives, batch_size // 2)))
                positives = positives[batch_size // 2:]
            else:
                # for the last batch where no. of positive could be < batch_size
                temp = np.concatenate((positives, np.random.choice(negatives, len(positives))))
                positives = np.array([])
            np.random.shuffle(temp)
            task_batch.append(temp)
        return np.asarray(task_batch)


def custom_bce(y, p, offset=1e-7):  # 1e-10
    #p_ = p
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = -tf.reduce_mean(p * tf.math.log(y) + (1.0 - p_) * tf.math.log(1.0 - y), axis=1)
    return tf.reduce_mean(vec_bce) / y.shape[0]

# def custom_bce(y, t):
#     return - tf.reduce_mean(
#         tf.multiply(t, tf.math.log(y)) + tf.multiply((1-t), tf.math.log(1-y)))

def custom_KLD(q, p, offset=1e-7):  # q is model predictive/approximating distribution, p is target distribution
    q_ = tf.clip_by_value(q, offset, 1 - offset)
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = tf.reduce_sum(p_ * (tf.math.log(p_) - tf.math.log(q_)))
    return tf.reduce_sum(vec_bce) / p.shape[0]


class Network(object):

    def __init__(self, n_layers):
        self.params = []
        self.W1 = tf.Variable(
            tf.random.normal([n_layers[0], n_layers[1]], stddev=0.1),
            name='W1')
        #         self.b1 = tf.Variable(tf.random.normal([n_layers[1]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b1')
        self.b1 = tf.Variable(tf.zeros([1, n_layers[1]]))
        self.W2 = tf.Variable(
            tf.random.normal([n_layers[1], n_layers[2]], stddev=0.1),
            name='W2')
        #         self.b2 = tf.Variable(tf.random.normal([n_layers[2]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b2')
        self.b2 = tf.Variable(tf.zeros([1, n_layers[2]]))
        self.W3 = tf.Variable(
            tf.random.normal([n_layers[2], n_layers[3]],stddev=0.1),
            name='W3')
        #         self.b3 = tf.Variable(tf.random.normal([n_layers[3]], mean=0.0, stddev=0.1, dtype=tf.dtypes.float32, seed=0), name='b3')
        self.b3 = tf.Variable(tf.zeros([1, n_layers[3]]))
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        # self.params.append(self.W1)
        # self.params.append(self.b1)
        # self.params.append(self.W2)
        # self.params.append(self.b2)
        # self.params.append(self.W3)
        # self.params.append(self.b3)

    def forward(self, x):
        X_tf = tf.cast(x, dtype=tf.float32)
        Z1 = tf.matmul(X_tf, self.W1) + self.b1
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.W2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.W3) + self.b3
        Y = tf.nn.sigmoid(Z3)
        return Y

    def loss(self, y_true , y_pred):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        y_pred_tf = tf.cast(tf.reshape(y_pred, (-1, 1)), dtype=tf.float32)
        return tf.reduce_mean(tf.compat.v1.losses.log_loss(y_true_tf, y_pred_tf))

    def backward(self, x,y):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        #print(type(y))
        # try:
        #     y = y.numpy().reshape((50,))
        # except:
        #     y = y.numpy().reshape((14,))
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.GradientTape() as tape:
            predicted = self.forward(x)
            #predicted = predicted.numpy().reshape((50,))
            current_loss = self.loss(y, predicted)
        #print(current_loss)
        grads = tape.gradient(current_loss, self.params)
        #print(grads)
        optimizer.apply_gradients(zip(grads, self.params),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())


trainX = pd.read_csv('./data/mnist/trainX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
trainY = pd.read_csv('./data/mnist/trainY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
testX = pd.read_csv('./data/mnist/testX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
testY = pd.read_csv('./data/mnist/testY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
validX = pd.read_csv('./data/mnist/validX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
validY = pd.read_csv('./data/mnist/validY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()

trainY = relabel(trainY)
testY = relabel(testY)
validY = relabel(validY)
print(trainY.shape)
print(validY.shape)
train_losses = {}
valid_losses = {}
n_tasks = 2
n_epochs = 1
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
net = Network([784, 312, 128, 1])

tqdm.write('Running Tasks')
for task in tqdm(range(n_tasks)):
    train_losses[task] = []
    valid_losses[task] = []

    for epoch in range(n_epochs):
        batches = batch_loader(trainY[task], 50, class_bal=True)
        for batch in batches:
            preds = net.forward(trainX[batch])
            batch_loss = net.loss(trainY[task, batch], preds)
            print("batch loss = .%5f" % batch_loss)
            net.backward(trainX[batch],trainY[task, batch])
            # print(net.backward.l)

            # with tf.GradientTape() as tape:
            #     output = net.forward(trainX[batch])
            #     #print(output)
            #     #batch_loss = custom_bce(trainY[task, batch], output)
            #     try:
            #         output = output.numpy().reshape((50,))
            #     except:
            #         output = output.numpy().reshape((14,))
            #     batch_loss = tf.compat.v1.losses.log_loss(trainY[task, batch], output)
            #     print(batch_loss)
            #     #break
            #     # output = output.numpy().reshape((50,))
            #     #batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=trainY[task, batch], logits=output, name=None)
            # #             print('batch loss:', batch_loss)
            #
            # grads = tape.gradient(batch_loss, net.params)
            # optimizer.apply_gradients(zip(grads, net.params))

        train_loss = 0
        size = 0

        batches = batch_loader(trainY[task], 50, class_bal=True)
        for batch in batches:
            size += batch.shape[0]
            output = net.forward(trainX[batch])
            # try:
            #     output = output.numpy().reshape((50,))
            # except:
            #     output = output.numpy().reshape((14,))
            train_loss = net.loss(trainY[task, batch], output) * batch.shape[0]
            #         print('train output:', output)
            # train_loss += custom_bce(trainY[task, batch], output) * batch.shape[0]
        train_losses[task].append(train_loss / size)

        valid_loss = 0
        size = 0

        batches = batch_loader(validY[task], 50, class_bal=True)
        # for batch in batches:
        #     size += batch.shape[0]
        #     # output = net.forward(validX[batch])
        #     # try:
        #     #     output = output.numpy().reshape((50,))
        #     # except:
        #     #     output = output.numpy().reshape((14,)
        #     valid_loss = net.loss(validY[task, batch], output) * batch.shape[0]
        #     # try:
        #     #
        #     # except:
        #
        #     # valid_loss += custom_bce(validY[task, batch], output) * batch.shape[0]
        # valid_losses[task].append(valid_loss / size)

print('Experiment complete')

# In[17]:


# get_ipython().run_line_magic('matplotlib', 'notebook')
print(train_losses[0])
# plt.plot(train_losses[0])
# plt.plot(valid_losses[0])
plt.show()
