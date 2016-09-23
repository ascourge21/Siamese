import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Lambda, Dense, Dropout
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
    seq.add(Dense(128, input_shape=(input_d,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


# get the data
samp_f = 3
total_to_samp = 20000
x, y = createFaceData.gen_data_new(samp_f, total_to_samp)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim = x_train.shape[2]
input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))
base_network = create_base_network(input_dim)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
nb_epoch = 15
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([x_train[:, 0], x_train[:, 1]])
tr_acc = compute_accuracy(pred, y_train)
pred = model.predict([x_test[:, 0], x_test[:, 1]])
te_acc = compute_accuracy(pred, y_test)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
