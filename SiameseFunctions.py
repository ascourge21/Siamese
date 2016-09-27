"""
    helper functions to build Siamese network and cost functions.
"""

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import backend as K

# from keras.regularizers import WeightRegularizer, l2
# from keras.layers.normalization import BatchNormalization
import numpy as np


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels, thresh):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < thresh].mean()


def create_base_network(input_d, hidden_layer_size):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    for i in range(len(hidden_layer_size)):
        if i == 0:
            seq.add(Dense(hidden_layer_size[i], input_shape=(input_d,), activation='linear'))
        else:
            seq.add(Dense(hidden_layer_size[i], activation='linear'))
        seq.add(Dropout(0.2))
    return seq

