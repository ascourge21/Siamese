"""
    from here on, we'll not create separate endo/epi model.
    These models will be based on good common practice from literature
    Ex: http://www.cc.gatech.edu/~hays/compvision/proj6/

    variations shall be made in maxpool vs no maxpool etc
    batch_size shall be 64/128 - literature seems to suggest 128 is good?

    - this is for 9X9X9 patches
    - 2 convolution layers of channels - 3, 9 and conv kernel sized 3X3X3
    - dense layer try 50 or 100?
"""

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Convolution3D, \
    Flatten, BatchNormalization
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc

from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss
from siamese_supervised import createShapeData


# a CNN layer for intensity inputs
def cnn_block_without_maxpool(input_dim, no_conv_filt, dense_n):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layers
    # layer 1
    conv_channel_1 = 5
    seq.add(Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu', subsample=(2, 2, 2)))
    # seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    # layer 2
    conv_channel_2 = 15
    seq.add(Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    # dense layers - one or two?
    seq.add(Flatten())
    seq.add(Dense(75, activation='relu'))
    seq.add(BatchNormalization(mode=2))
    seq.add(Dense(30, activation='relu'))
    seq.add(BatchNormalization(mode=2))
    return seq


# train model given x_train and y_train
def train_model(x_tr, y_tr, conv_f_n, dense_n):
    save_name = 'shape_match_model1.h5'
    tr_epoch = 10

    input_dim = x_tr.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = cnn_block_without_maxpool(input_dim, conv_f_n, dense_n)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model_tr = Model(input=[input_a, input_b], output=distance)

    # train
    # opt_func = RMSprop(lr=.0005, clipnorm=1)
    opt_func = RMSprop()
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    model_tr.fit([x_tr[:, 0], x_tr[:, 1]], y_tr, validation_split=.30,
                 batch_size=128, verbose=2, nb_epoch=tr_epoch,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model_tr.save(save_name)
    return model_tr


# test, also provide info on which pair it was trained on and which it was tested on
def run_test(model, x_ts, y_ts, tr_ids, ts_n, conv_n, dense_n):
    # compute final accuracy on training and test sets
    pred_ts = model.predict([x_ts[:, 0], x_ts[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_ts, pred_ts)
    roc_auc = auc(fpr, tpr)
    target = open('auc_scores_summary_model1.txt', 'a')
    target.write("endo, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")
    target.close()
    print("endo, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")


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


# load 1 and 2 and test on 3
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
# src = '/home/nripesh/Dropbox/temp_images/run_on_allens/'
data_stem = 'x_data_intensity_comb_'


# run this to perform cross validation
def do_cross_val():
    conv_n_vals = [15]
    dense_n_vals = [100]
    avail_ids = [1, 2, 3, 4, 5]

    for conv_n in conv_n_vals:
        for dense_n in dense_n_vals:
            for idi in avail_ids:
                # test on idi, train on all except idi
                test_id = idi
                tr_id = [i for i in avail_ids if i != idi]

                x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)
                model = train_model(x_train, y_train, conv_n, dense_n)
                run_test(model, x_test, y_test, tr_id, test_id, conv_n, dense_n)
                print()


# run this to get the final model
def train_final_model():
    conv_n = 15
    dense_n = 50
    tr_id = [1, 3, 4]
    test_id = 2
    x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)
    model = train_model(x_train, y_train, conv_n, dense_n)
    run_test(model, x_test, y_test, tr_id, test_id, conv_n, dense_n)
    print("endo, trained on: " + str(tr_id) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + "\n")


# do_cross_val()
train_final_model()
