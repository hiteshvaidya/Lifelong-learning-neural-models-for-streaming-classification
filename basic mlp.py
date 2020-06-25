#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle as pkl


# In[10]:


def relabel(labels):
    '''
    task wise relabel the dataset as combination of 0's and 1's
    '''
    new_labels = np.empty((0, labels.shape[0]), dtype=np.float32)
    for task in range(10):
        positives = np.where(labels[:,task] == 1)[0]
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
        positives = np.where(labels==1)[0]
        negatives = np.arange(labels.shape[0])
        negatives = np.delete(negatives, positives)
        np.random.shuffle(negatives)
        np.random.shuffle(positives)
        task_batch = []
        # create batches by iteratively scraping out chunks out of positives array
        while positives.shape[0] > 0:
            if len(positives) >= batch_size/2:
                # create a batch such that positive (batch_size/2) is added with sampled negatives (batch_size/2)
                temp = np.concatenate((positives[:batch_size//2], np.random.choice(negatives, batch_size//2)))
                positives = positives[batch_size//2: ]
            else:
                # for the last batch where no. of positive could be < batch_size
                temp = np.concatenate((positives, np.random.choice(negatives, len(positives))))
                positives = np.array([])
            np.random.shuffle(temp)
            task_batch.append(temp)
        return np.asarray(task_batch)

def custom_bce(y, p, offset=1e-7): #1e-10
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = -tf.reduce_sum(y * tf.math.log(p_) + (1.0 - y) * tf.math.log(1.0 - p_), axis=1)
    return tf.reduce_sum(vec_bce)/y.shape[0]

def custom_KLD(q, p, offset=1e-7): # q is model predictive/approximating distribution, p is target distribution
    q_ = tf.clip_by_value(q, offset, 1 - offset)
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = tf.reduce_sum(p_ * (tf.math.log(p_) - tf.math.log(q_)))
    return tf.reduce_sum(vec_bce)/p.shape[0]

class Network(object):
    
    def __init__(self, n_layers):
        self.params = []
        self.W1 = tf.Variable(tf.random.normal([n_layers[0], n_layers[1]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='W1')
        self.b1 = tf.Variable(tf.random.normal([n_layers[1]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='b1')
        self.W2 = tf.Variable(tf.random.normal([n_layers[1], n_layers[2]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='W2')
        self.b2 = tf.Variable(tf.random.normal([n_layers[2]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='b2')
        self.W3 = tf.Variable(tf.random.normal([n_layers[2], n_layers[3]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='W3')
        self.b3 = tf.Variable(tf.random.normal([n_layers[3]], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=0), name='b3')
        
        self.params.append(self.W1)
        self.params.append(self.b1)
        self.params.append(self.W2)
        self.params.append(self.b2)
        self.params.append(self.W3)
        self.params.append(self.b3)
        
    def forward(self, x):
        Z1 = tf.matmul(x, self.W1) + self.b1
        Z1 = tf.nn.relu(Z1)
        Z2 = tf.matmul(Z1, self.W2) + self.b2
        Z2 = tf.nn.relu(Z2)
        Z3 = tf.matmul(Z2, self.W3) + self.b3
        Y = tf.nn.sigmoid(Z3)
        return Y
    

trainX = pd.read_csv('datasets/mnist/trainX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
trainY = pd.read_csv('datasets/mnist/trainY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
testX = pd.read_csv('datasets/mnist/testX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
testY = pd.read_csv('datasets/mnist/testY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
validX = pd.read_csv('datasets/mnist/validX.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()
validY = pd.read_csv('datasets/mnist/validY.tsv', sep="\t", header=None, index_col=False, dtype=np.float32).to_numpy()

trainY = relabel(trainY)
testY = relabel(testY)
validY = relabel(validY)

train_losses = {}
valid_losses = {}
n_tasks = 2
n_epochs = 5
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
net = Network([784,312,128,1])

tqdm.write('Running Tasks')
for task in tqdm(range(n_tasks)):
    batches = batch_loader(trainY[task], 50, class_bal=True)
    for batch in batches:
        with tf.GradientTape() as tape:
            output = net.forward(trainX[batch])
            batch_loss = custom_bce(trainY[task, batch], output)
#         print('batch loss:', batch_loss)
        grads = tape.gradient(batch_loss, net.params)
        optimizer.apply_gradients(zip(grads, net.params))

    train_loss = 0
    size = 0
    train_losses[task] = []
    batches = batch_loader(trainY[task], 50, class_bal=True)
    for batch in batches:
        size += batch.shape[0]
        output = net.forward(trainX[batch])
#         print('train output:', output)
        train_loss += custom_bce(trainY[task, batch], output) * batch.shape[0]
    train_losses[task].append(train_loss/size)
    
    valid_loss = 0
    size = 0
    valid_losses[task] = []
    batches = batch_loader(validY[task], 50, class_bal=True)
    for batch in batches:
        size += batch.shape[0]
        output = net.forward(validX[batch])
        valid_loss += custom_bce(validY[task, batch], output) * batch.shape[0]
    valid_losses[task].append(valid_loss/size)
        
        
print('Experiment complete')
        


# In[11]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.plot(train_losses[0])
plt.plot(valid_losses[0])
plt.show()

