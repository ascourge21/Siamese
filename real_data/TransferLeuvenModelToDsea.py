"""
    Here we'll attempt at transferring the simulated model's weights and train top layers
    with the cardiac data.

    This uses code from the more recent Train/test simulation models.

    # to run directly on terminal
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 TransferLeuvenModelToDsea.py
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Convolution3D, \
    Flatten, BatchNormalization
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc

from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss
from siamese_supervised import createShapeData


# a CNN layer for intensity inputs
def create_cnn_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    # conv layers
    kern_size = 3
    seq.add(Convolution3D(5, kern_size, kern_size, kern_size, input_shape=input_dim,  # subsample=(2, 2, 2),
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.25))
    seq.add(BatchNormalization(mode=2))

    kern_size = 3
    seq.add(Convolution3D(10, kern_size, kern_size, kern_size,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.25))
    seq.add(BatchNormalization(mode=2))

    kern_size = 3
    seq.add(Convolution3D(20, kern_size, kern_size, kern_size,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.25))
    seq.add(BatchNormalization(mode=2))

    # dense layers
    dense_n = 50
    seq.add(Flatten())
    seq.add(Dense(dense_n, activation='relu'))
    seq.add(Dropout(.25))
    seq.add(BatchNormalization(mode=2))
    return seq


# create groups of 4 image sets as training and 1 as test
def train_from_leuven_data():
    src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
    data_stem = 'x_data_intensity_comb_'
    save_name = 'leuven_model_to_transfer_k3.h5'
    tr_epoch = 5

    x_tr = []
    y_tr = []
    train_ids = [1, 2, 3, 4, 5]
    test_id = 2
    for tid in train_ids:
        train_name = data_stem + str(tid)
        x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all = np.concatenate(x_tr)
    y_tr_all = np.concatenate(y_tr)

    # test data
    test_name = data_stem + str(test_id)
    x_test, y_test = createShapeData.get_int_paired_format(src, test_name)

    input_dim = x_tr_all.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = create_cnn_network(input_dim)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model_tr = Model(input=[input_a, input_b], output=distance)

    # train
    opt_func = RMSprop()
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    model_tr.fit([x_tr_all[:, 0], x_tr_all[:, 1]], y_tr_all, validation_split=.30,
                 batch_size=128, verbose=2, nb_epoch=tr_epoch,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model_tr.save('/home/nripesh/PycharmProjects/Siamese/real_data/' + save_name)

    # test
    # compute final accuracy on training and test sets
    pred_ts = model_tr.predict([x_test[:, 0], x_test[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_test, pred_ts)
    roc_auc = auc(fpr, tpr)
    target = open('auc_scores_summary_transfer_learning.txt', 'a')
    target.write("endo, trained on: " + str(train_ids) + ", tested on: " + str(test_id) + ", auc: " + str(roc_auc) + "\n")
    target.close()
    print("endo, trained on: " + str(train_ids) + ", tested on: " + str(test_id) + ", auc: " + str(roc_auc) + "\n")

    return model_tr


# now train on dsea_data
def train_on_dsea_data(model):
    # first freeze if necessary
    to_freeze = False
    if to_freeze:
        # freeze the feature generation layers - no need to train these.
        no_to_freeze = 2
        for i in range(no_to_freeze):
            model.layers[2].layers[i].Trainable = False
            print('frozen: ' + str(model.layers[2].layers[i]))

    # load dsea patch data
    dsea_src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/' \
               'dsea_data_based_train_patches/'
    data_dsea_name = 'dsea_data_patch_pairs_augm_size_9'
    save_dsea_name = 'dsea_trf_and_augm_match_model_k3.h5'

    x_d, y_d = createShapeData.get_int_paired_format(dsea_src, data_dsea_name)
    x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(x_d, y_d, test_size=.25)

    # compile and train again
    nb_epoch_dsea = 15
    opt_func = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=opt_func)
    model.fit([x_train_d[:, 0], x_train_d[:, 1]], y_train_d, validation_split=.25,
              batch_size=32, verbose=2, nb_epoch=nb_epoch_dsea,
              callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model.save('/home/nripesh/PycharmProjects/Siamese/real_data/' + save_dsea_name)

    # compute final accuracy on training and test sets
    pred_tr = model.predict([x_train_d[:, 0], x_train_d[:, 1]])
    pred_ts = model.predict([x_test_d[:, 0], x_test_d[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_test_d, pred_ts)
    roc_auc = auc(fpr, tpr)
    print('AUC score: ' + str(roc_auc))


# model from Leuven
model = train_from_leuven_data()
train_on_dsea_data(model)