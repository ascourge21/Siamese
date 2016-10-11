import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.layers import Activation
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras import backend as K

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

import createFaceData


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


def create_base_network(input_d):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    nb_filter = [6, 12]
    kern_size = 3
    # conv layers
    # seq.add(Reshape((1, 38, 31), input_shape=(38, 31)))
    seq.add(Convolution2D(nb_filter[0], kern_size, kern_size, input_shape=input_d,
                          border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
    seq.add(Dropout(.25))
    # conv layer 2
    seq.add(Convolution2D(nb_filter[1], kern_size, kern_size, border_mode='valid', dim_ordering='th'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))  # downsample
    seq.add(Dropout(.25))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


# get the data
samp_f = 2
total_to_samp = 10000
x, y = createFaceData.gen_train_data_for_conv_new(samp_f, total_to_samp)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim = x_train.shape[2:]
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)
base_network = create_base_network(input_dim)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
nb_epoch = 15
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
xtr1 = x_train[:, 0]
xtr2 = x_train[:, 1]
model.fit([xtr1, xtr2], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([x_train[:, 0], x_train[:, 1]])
tr_acc = compute_accuracy(pred, y_train)
pred = model.predict([x_test[:, 0], x_test[:, 1]])
te_acc = compute_accuracy(pred, y_test)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
