'''
Contains code for both left and right network merged together

@author: Hitesh Vaidya
'''

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt


class Network():

    def __init__(self, n_x, n_z1, n_z2, n_y):
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

        # now declare the weights connecting the input to the hidden layer
        self.W1 = tf.Variable(
            tf.random.normal([n_x, n_z1],mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='W1')
        self.b1 = tf.Variable(
            tf.random.normal([1, n_z1], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0),  name='b1')
        self.W2 = tf.Variable(
            tf.random.normal([n_z1, n_z2], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='W2')
        self.b2 = tf.Variable(tf.random.normal([1, n_z2], mean=0.0, stddev=0.1,
                                               dtype=tf.dtypes.float32, seed=0),
                              name='b2')
        self.W3 = tf.Variable(
            tf.random.normal([n_z2, n_y], mean=0.0, stddev=0.1,
                             dtype=tf.dtypes.float32, seed=0), name='W3')
        self.b3 = tf.Variable(tf.random.normal([1, n_y], mean=0.0, stddev=0.1,
                                               dtype=tf.dtypes.float32, seed=0),
                              name='b3')

        self.params.append(self.m1)
        self.params.append(self.n1)
        self.params.append(self.m2)
        self.params.append(self.n2)
        self.params.append(self.m21)
        self.params.append(self.n21)
        self.params.append(self.m10)
        self.params.append(self.n10)

        self.params.append(self.W1)
        self.params.append(self.b1)
        self.params.append(self.W2)
        self.params.append(self.b2)
        self.params.append(self.W3)
        self.params.append(self.b3)

    def forward(self, x, tau=0.0):
        Z1 = tf.matmul(x, self.m1) + self.n1
        Z1 = tf.nn.sigmoid(Z1)
        Z2 = tf.matmul(Z1, self.m2) + self.n2
        Z2 = tf.nn.sigmoid(Z2)
        Z1_hat = tf.matmul(Z2, self.m21) + self.n21
        Z1_hat = tf.nn.sigmoid(Z1_hat)
        Z0_hat = tf.matmul(Z1, self.m10) + self.n10
        Z0_hat = tf.nn.sigmoid(Z0_hat)

        H1 = tf.matmul(x, self.W1) + self.b1
        H1 = tf.nn.relu(H1) * (tau * Z1 + (1-tau))
        H2 = tf.matmul(H1, self.W2) + self.b2
        H2 = tf.nn.relu(H2) * (tau * Z2 + (1-tau))
        Y = tf.matmul(H2, self.W3) + self.b3
        Y = tf.nn.sigmoid(Y)

        at_loss = custom_MSE(x, Z0_hat) + custom_MSE(Z1, Z1_hat)
        return Y, at_loss

    def loss(self, y_true, y_pred, choice='log'):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, 1)), dtype=tf.float32)
        y_pred_tf = tf.cast(tf.reshape(y_pred, (-1, 1)), dtype=tf.float32)
        if choice == 'log':
            return tf.reduce_mean(
                tf.compat.v1.losses.log_loss(y_true_tf, y_pred_tf))
        elif choice == 'bce':
            return custom_bce(y_true_tf, y_pred_tf)
        elif choice == 'mse':
            return custom_MSE(y_true_tf, y_pred_tf)

    def backward(self, x, y, tau, choice):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=0.01)
        with tf.GradientTape() as tape:
            predicted, left_network_loss = self.forward(x, tau)
            current_loss = self.loss(y, predicted, choice) + tau * \
                           left_network_loss
        grads = tape.gradient(current_loss, self.params)
        optimizer.apply_gradients(zip(grads, self.params),
                                  global_step=tf.compat.v1.train.get_or_create_global_step())

# download mnist data from tensorflow
(ds_train, ds_test) = tfds.as_numpy(tfds.load(
                                    'mnist',
                                    split=['train', 'test'],
                                    shuffle_files=True,
                                    as_supervised=True,
                                    batch_size=-1
                                    ))

def normalize_img(images):
    '''
    Flatten and Normalizes images: `uint8` -> `float32`
    :param images: input images in data
    :return: normalized vector
    '''
    # images = tf.reshape(images, [images.shape[0], -1])
    return tf.cast(images, tf.float32) / 255.0


def relabel(labels):
    '''
    task wise relabel the dataset as combination of 0's and 1's
    '''
    new_labels = np.empty((0, labels.shape[0]), dtype=np.float32)
    for task in range(10):
        positives = np.where(labels == task)[0]
        task_labels = np.zeros(labels.shape[0], dtype=np.float32)
        task_labels[positives] = 1.0
        new_labels = np.vstack((new_labels, task_labels))
#     new_labels = tf.convert_to_tensor(new_labels, dtype=tf.float32)
    return new_labels


# separate images and labels
trainX = ds_train[0]
trainY = ds_train[1]
testX = ds_test[0]
testY = ds_test[1]

trainX = np.reshape(trainX, (trainX.shape[0], -1))
testX = np.reshape(testX, (testX.shape[0], -1))

validX = np.empty((0,784))
validY = np.array([])
for num in range(10):
    indices = np.where(trainY == num)[0]
    indices = indices[:int(0.1 * len(indices))]
    validX = np.append(validX, trainX[indices], axis=0)
    validY = np.append(validY, trainY[indices])
    trainX = np.delete(trainX, indices, axis=0)
    trainY = np.delete(trainY, indices)

indices = np.arange(validX.shape[0])
np.random.shuffle(indices)
validX = validX[indices]
validY = validY[indices]
print('train shapes:', trainX.shape, trainY.shape)
print('valid shapes:', validX.shape, validY.shape)
print('test shapes:', testX.shape, testY.shape)

# normalize images
trainX = normalize_img(trainX)
testX = normalize_img(testX)
validX = normalize_img(validX)

trainY = relabel(trainY)
testY = relabel(testY)
validY = relabel(validY)

# Declare the network
net = Network(784, 312, 128, 1)
n_tasks = 10
n_epochs = 5
batch_size = 50
tau = 0.01

# train_losses[t] = [n_epochs x n_tasks]
train_losses = {}
test_losses = {}
valid_losses = {}
train_accuracy = {}
valid_accuracy = {}
test_accuracy = {}
train_autoencoder_loss = {}
valid_autoencoder_loss = {}
test_autoencoder_loss = {}
for t in range(n_tasks):
    train_losses[t] = []
    valid_losses[t] = []
    test_losses[t] = []
    train_accuracy[t] = []
    valid_accuracy[t] = []
    test_accuracy[t] = []
    train_autoencoder_loss[t] = []
    valid_autoencoder_loss[t] = []
    test_autoencoder_loss[t] = []

def custom_bce(y_true, y_pred, offset=1e-7):  # 1e-10
    '''
    calculates Binary Cross Entropy value
    :param y_true: true labels
    :param y_pred: predicted output
    :param offset: threshold for decimal value
    :return: BCE loss value
    '''
    y_true = tf.clip_by_value(y_true, offset, 1 - offset)
    y_pred = tf.clip_by_value(y_pred, offset, 1 - offset)
    # p_ = tf.clip_by_value(p, offset, 1 - offset)
    vec_bce = -tf.reduce_mean(
        y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 -
                                                                    y_pred),
        axis=1)
    return tf.reduce_mean(vec_bce)

def custom_KLD(y_true, y_pred, offset=1e-7):  # q is model
    # predictive/approximating
    # distribution, p is target distribution
    '''
    Calculates KL-Divergence
    :param y_true: true labels
    :param y_pred: predicted output
    :param offset: threshold for decimal value
    :return: KL-Divergence
    '''
    y_true = tf.clip_by_value(y_true, offset, 1 - offset)
    y_pred = tf.clip_by_value(y_pred, offset, 1 - offset)
    vec_bce = tf.reduce_sum(y_true * (tf.math.log(y_true) - tf.math.log(y_pred)),
                            axis=-1)
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

def batch_loader(labels, batch_size, class_bal=False):
    '''
    Generate random batches of data
    :param labels: expected output data
    :param batch_size: batch size
    :param class_bal: Flag for class balance
    :return: None
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
                temp = np.concatenate((positives[:batch_size // 2],
                                       np.random.choice(negatives,
                                                        batch_size // 2)))
                positives = positives[batch_size // 2:]
            else:
                # for the last batch where no. of positive could be < batch_size
                temp = np.concatenate(
                    (positives, np.random.choice(negatives, len(positives))))
                positives = np.array([])
            np.random.shuffle(temp)
            task_batch.append(temp)
        return np.asarray(task_batch)


def train_metrics():
    '''
    Calculates train loss and accuracy
    :return: None
    '''
    for t in range(n_tasks):
        loss = 0
        train_acc = 0
        loss_at = 0
        size = 0
        batches = batch_loader(trainY[t], batch_size, class_bal=True)
        for batch in batches:
            output, autoencoder_loss = net.forward(tf.gather(trainX, batch))
            size += batch.shape[0]
            temp = tf.convert_to_tensor(trainY[t, batch].reshape(
                output.shape), dtype=tf.float32)
            loss += (net.loss(temp, output, 'bce') + tau*autoencoder_loss) * \
                    batch.shape[0]
            loss_at += autoencoder_loss * batch.shape[0]
            output = output.numpy().reshape(-1)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            train_acc += np.sum(output == trainY[t, batch])

        train_losses[t].append(loss / size)
        train_accuracy[t].append(train_acc / size)
        train_autoencoder_loss[t].append(loss_at / size)

def valid_metrics():
    '''
    Calculate validation loss and accuracy
    :return: None
    '''
    for t in range(n_tasks):
        valid_loss = 0
        valid_acc = 0
        loss_at = 0
        size = 0
        batches = batch_loader(validY[t], batch_size, class_bal=True)
        for batch in batches:
            output, autoencoder_loss = net.forward(tf.gather(validX, batch))
            size += batch.shape[0]
            temp = tf.convert_to_tensor(validY[t, batch].reshape(
                output.shape), dtype=tf.float32)
            valid_loss += (net.loss(temp, output, 'bce') + tau *
                           autoencoder_loss) * batch.shape[0]
            loss_at += autoencoder_loss * batch.shape[0]
            output = output.numpy().reshape(-1)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            valid_acc += np.sum(output == validY[t, batch])

        valid_losses[t].append(valid_loss/size)
        valid_accuracy[t].append(valid_acc/size)
        valid_autoencoder_loss[t].append(loss_at/size)

def test_metrics():
    '''
    Calculate test loss and accuracy
    :return: None
    '''
    for t in range(n_tasks):
        test_loss = 0
        test_acc = 0
        loss_at = 0
        size = 0
        batches = batch_loader(testY[t], batch_size, class_bal=True)
        for batch in batches:
            output, autoencoder_loss = net.forward(tf.gather(testX, batch))
            size += batch.shape[0]
            temp = tf.convert_to_tensor(testY[t, batch].reshape(
                output.shape), dtype=tf.float32)
            test_loss += (net.loss(temp, output, 'bce') + tau *
                          autoencoder_loss) * batch.shape[0]
            loss_at += autoencoder_loss * batch.shape[0]
            output = output.numpy().reshape(-1)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            test_acc += np.sum(output == testY[t, batch])

        valid_losses[t].append(test_loss / size)
        valid_accuracy[t].append(test_acc / size)
        valid_autoencoder_loss[t].append(loss_at / size)

train_metrics()
valid_metrics()

# train and calculate BWT, FWT
tqdm.write('Tasks running')

for task in tqdm(range(n_tasks)):
    tqdm.write('Epochs running')
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        batches = batch_loader(trainY[task], batch_size, class_bal=True)
        for batch in batches:
            net.backward(tf.gather(trainX, batch), tf.gather(trainY[task],
                                                             batch), tau, 'bce')

        #     loss_diff.append(tf.math.abs(log_loss - batch_loss))
        # temp = tf.math.reduce_mean(loss_diff)
        # compare_loss.append(temp)
        train_metrics()
        valid_metrics()
    test_metrics()

data_dict = {'train_loss': train_losses,
             'valid_loss': valid_losses,
             'test_loss': test_losses,
             'train_acc': train_accuracy,
             'valid_acc': valid_accuracy,
             'test_acc': test_accuracy,
             'train_autoencoder_loss': train_autoencoder_loss,
             'valid_autoencoder_loss': valid_autoencoder_loss,
             'test_autoencoder_loss': test_autoencoder_loss
             }

pkl.dump(data_dict, open('results/merged_data_dict_tau=1.pkl', 'wb'))


def plot_loss():
    '''
    plots task wise train and validation losses of the model
    :return: None
    '''
    fig, ax = plt.subplots(nrows=5, ncols=2)
    count = 0
    for r in range(5):
        for c in range(2):
            ax[r,c].plot(train_losses[count])
            ax[r,c].plot(valid_losses[count])
            count += 1
    plt.show()


def plot_test_loss():
    '''
    plots task wise test losses of the model
    :return: None
    '''
    fig, ax = plt.subplots(nrows=5, ncols=2)
    count = 0
    for r in range(5):
        for c in range(2):
            ax[r,c].plot(test_losses[count])
            count += 1
    plt.show()


def plot_accuracy():
    '''
    plots task wise train and validation accuracies of the model
    :return:
    '''
    fig, ax = plt.subplots(nrows=5, ncols=2)
    count = 0
    for r in range(5):
        for c in range(2):
            ax[r, c].plot(train_accuracy[count])
            ax[r, c].plot(valid_accuracy[count])
            count += 1
    plt.show()


plot_loss()
plot_test_loss()
plot_accuracy()

print('Test losses:')
print(test_losses)