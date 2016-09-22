import numpy as np
import pickle
import csv
from keras.utils import np_utils
from matplotlib import pyplot as plt


def load_mnist_train():
    mnist_train = []
    with open('data/mnist_train.csv', 'rt') as csvfile:
        cread = csv.reader(csvfile)
        for row in cread:
            vals = np.array([float(x) / 256 for x in row[1:]])
            vals = vals.reshape((784, 1))
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


def get_train_for_a_class(x_train, y_l, label):
    count = 0
    i = 0
    x_train_lbl = np.zeros([1000, 784])
    while count < 1000:
        if y_l[i] == label:
            x_train_lbl[count, :] = x_train[i, :]
            count += 1
        i += 1
    return x_train_lbl


def get_train_data(total_to_samp, total_classes):
    samp_f = 2
    x_train, y_train, y_l = load_mnist_train()
    # x_train = pickle.load(open("x_train.p", "rb"))
    # y_train = pickle.load(open("y_train.p", "rb"))
    # y_l = pickle.load(open("y_l.p", "rb"))

    total_per_class = 1000
    orig_dim = 784
    x_train_labeled = np.zeros([total_per_class, orig_dim, total_classes])
    for i in range(total_classes):
        x_train_labeled[:,:,i] = get_train_for_a_class(x_train, y_l, i)

    x_tr_m = np.zeros([total_to_samp, int(orig_dim/(samp_f*samp_f))])
    y_tr_m = np.zeros([total_to_samp, 1])
    count = 0
    for i in range(int(total_to_samp/(2*total_classes))):
        for j in range(total_classes):
            ind1 = 0
            ind2 = 0
            while True:
                ind1 = np.random.randint(total_per_class)
                ind2 = np.random.randint(total_per_class)
                if ind1 != ind2:
                    break
            # print(ind1)
            # print(ind2)

            x_a = x_train_labeled[ind1, :, j]
            x_b = x_train_labeled[ind2, :, j]


            # plt.figure(1)
            # plt.imshow(np.reshape(x_a, [28, 28]), cmap='Greys_r')
            # plt.figure(2)
            # plt.imshow(np.reshape(x_b, [28, 28]), cmap='Greys_r')
            # plt.show()

            if samp_f > 1:
                x_a = np.reshape(x_a, [28, 28])
                x_b = np.reshape(x_b, [28, 28])
                x_a = np.reshape(x_a[::samp_f, ::samp_f], [orig_dim/(samp_f*samp_f), ])
                x_b = np.reshape(x_b[::samp_f, ::samp_f], [orig_dim/(samp_f*samp_f), ])

            x_tr_m[count, :] = x_a
            y_tr_m[count] = 1
            count += 1
            x_tr_m[count, :] = x_b
            y_tr_m[count] = 1
            count += 1

    # total_to_samp = 32000
    x_tr_n = np.zeros([total_to_samp, int(orig_dim/(samp_f*samp_f))])
    y_tr_n = np.zeros([total_to_samp, 1])
    count = 0
    for i in range(int(total_to_samp/2)):
        ind1 = 0
        ind2 = 0
        while True:
            ind1 = np.random.randint(total_classes)
            ind2 = np.random.randint(total_classes)
            if ind1 != ind2:
                break
        # print(ind1)
        # print(ind2)

        # get data and reshape if necessary
        x_a = x_train_labeled[np.random.randint(total_per_class), :, ind1]
        x_b = x_train_labeled[np.random.randint(total_per_class), :, ind2]

        # plt.figure(1)
        # plt.imshow(np.reshape(x_a, [28, 28]), cmap='Greys_r')
        # plt.figure(2)
        # plt.imshow(np.reshape(x_b, [28, 28]), cmap='Greys_r')
        # plt.show()

        if samp_f > 1:
            x_a = np.reshape(x_a, [28, 28])
            x_b = np.reshape(x_b, [28, 28])
            x_a = np.reshape(x_a[::samp_f, ::samp_f], [orig_dim/(samp_f*samp_f), ])
            x_b = np.reshape(x_b[::samp_f, ::samp_f], [orig_dim/(samp_f*samp_f), ])

        x_tr_n[count, :] = x_a
        y_tr_n[count] = 0
        count += 1
        x_tr_n[count, :] = x_b
        y_tr_n[count] = 0
        count += 1

    x_train = np.concatenate([x_tr_m, x_tr_n], axis=0)
    y_train = np.concatenate([y_tr_m, y_tr_n], axis=0)

    return x_train, y_train
