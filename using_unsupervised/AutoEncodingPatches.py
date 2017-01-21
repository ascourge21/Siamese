"""
    - Here the idea is to use convolutional autoencoders to see if embeddings of intensity patches
    can be found
    - the best model (the encoding part) is also likely to work better in the supervised context
    - also transfer learning can be done to train in the supervised context or train on real data.
    - specially with real data - there's tonnes and tonnes of unlabeld patches so need to worry.

    - First try simple 1 or 2 dense layer model - baseline
    - Then try convolution + deeper models
"""


import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout, Convolution3D, \
    MaxPooling3D, Flatten, BatchNormalization, UpSampling3D
from keras.regularizers import WeightRegularizer, l2
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import train_test_split

import createShapeData
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss


# create groups of 4 image sets as training and 1 as test
def create_loo_train_test_set(src, data_stem, train_ids, test_id):
    x_tr = []
    y_tr = []
    for tid in train_ids:
        train_name = data_stem + str(tid)
        x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all = np.concatenate(x_tr)
    y_tr_all = np.concatenate(y_tr)

    test_name = data_stem + str(test_id)
    x_test, y_test = createShapeData.get_int_paired_format(src, test_name)
    return x_tr_all, x_test, y_tr_all, y_test

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'x_data_intensity_comb_'

tr_id = [1]
test_id = 2

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

x_train = x_train[:, 0]
x_test = x_test[:, 0]

input_dim = x_train.shape[1:]

# encoding layer
conv_channel_1 = 7
conv_channel_2 = 20
kern_size = 3
input_patches = Input(shape=input_dim)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(input_patches)
x = Dense(Dense(50, activation='relu'))(x)

# decoding layer
x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(encoded)
x = UpSampling3D(size=(2, 2, 2))(x)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(x)
x = UpSampling3D(size=(2, 2, 2))(x)
decoded = Convolution3D(1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                        border_mode='same')(x)

# compile and fit model
autoencoder = Model(input_patches, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                nb_epoch=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
