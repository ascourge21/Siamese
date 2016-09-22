import numpy as np
import pickle
import csv
from keras.datasets import cifar10
from keras.utils import np_utils


class DataLoader(object):
    @staticmethod
    def load_mnist_train():
        mnist_train = []
        with open('data/mnist_train.csv', 'rt') as csvfile:
            cread = csv.reader(csvfile)
            for row in cread:
                vals = np.array([float(x) / 256 for x in row[1:]])
                vals = vals.reshape((784,1))
                res = np.zeros((10, 1))
                res[int(row[0])] = 1
                mnist_train.append([vals, res, row[0]])
        x = np.zeros((len(mnist_train), 784))
        y = np.zeros((len(mnist_train), 10))
        y_l = np.zeros(len(mnist_train))
        for i in range(len(mnist_train)):
            x[i, :] = mnist_train[i][0].T
            y[i, :] = mnist_train[i][1].T
            y_l[i] = mnist_train[i][2]
        return x, y, y_l

    @staticmethod
    def load_mnist_test():
        mnist_test = []
        with open('data/mnist_test.csv', 'rt') as csvfile:
            cread = csv.reader(csvfile)
            for row in cread:
                vals = np.array([float(x) / 256 for x in row[1:]])
                vals = vals.reshape((784,1))
                res = np.int64(row[0])
                mnist_test.append([vals, res, row[0]])
            x = np.zeros((len(mnist_test), 784))
            y = np.zeros((len(mnist_test), 10))
            y_l = np.zeros(len(mnist_test))
            for i in range(len(mnist_test)):
                x[i, :] = mnist_test[i][0].T
                y[i, :] = mnist_test[i][1].T
                y_l[i] = mnist_test[i][2]
            return x, y, y_l

    @staticmethod
    def load_cifar():
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        # take only first 3 classes
        X_train = X_train[(y_train < 3).reshape(50000)]
        y_train = y_train[(y_train < 3).reshape(50000)]
        X_test = X_test[(y_test < 3).reshape(10000)]
        y_test = y_test[(y_test < 3).reshape(10000)]
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, 3)
        Y_test = np_utils.to_categorical(y_test, 3)
        return X_train, Y_train, X_test, Y_test

