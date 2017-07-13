"""
    this provides code to do cross validation on the conv/dense size and run the final model
    Multi-res (one has conv, smaller one has no conv)
    EPI
"""
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Convolution3D, Flatten, merge, BatchNormalization
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc

from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss
from siamese_supervised import createShapeData


# a CNN layer for intensity inputs
def create_cnn_network(input_dim, no_conv_filt, dense_n):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layer
    seq.add(Convolution3D(no_conv_filt, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    #seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    # dense layer
    seq.add(Flatten())
    seq.add(Dense(dense_n, activation='relu'))
    seq.add(BatchNormalization(mode=2))

    return seq


# a CNN layer for intensity inputs
def create_cnn_network_small(input_dim, no_conv_filt, dense_n):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layer
    seq.add(Convolution3D(no_conv_filt, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.2))
    seq.add(BatchNormalization(mode=2))

    # dense layer
    seq.add(Flatten())
    seq.add(Dense(dense_n, activation='relu'))
    seq.add(Dropout(.2))
    seq.add(BatchNormalization(mode=2))

    return seq


# train model given x_train and y_train
def train_model(x_tr_lg, y_train, x_tr_sm, conv_n, dense_n, save_name):
    nb_epoch = 5

    # will be shared across the two branches
    input_dim = x_tr_lg.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)

    input_dim2 = x_tr_sm.shape[2:]
    input_c = Input(shape=input_dim2)
    input_d = Input(shape=input_dim2)

    # the layer that takes larger patches - fix these for now and cval for the other stream
    conv_n_large = 15
    dense_n_large = 100
    cnn_network = create_cnn_network(input_dim, conv_n_large, dense_n_large)
    processed_a = cnn_network(input_a)
    processed_b = cnn_network(input_b)

    # the layer that takes smaller patches
    dense_network = create_cnn_network_small(input_dim2, conv_n, dense_n)
    processed_c = dense_network(input_c)
    processed_d = dense_network(input_d)

    # merge dense and cnn
    merged_a = merge([processed_a, processed_c], mode='concat', concat_axis=1)
    merged_b = merge([processed_b, processed_d], mode='concat', concat_axis=1)

    # custom distance
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([merged_a, merged_b])

    # the model, finally
    model_tr = Model(input=[input_a, input_b, input_c, input_d], output=distance)

    # train
    opt_func = RMSprop()
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    model_tr.fit([x_tr_lg[:, 0], x_tr_lg[:, 1], x_tr_sm[:, 0], x_tr_sm[:, 1]], y_train, validation_split=.30,
                 batch_size=128, verbose=2, nb_epoch=nb_epoch, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model_tr.save(save_name)
    return model_tr


# test, also provide info on which pair it was trained on and which it was tested on
def run_test(model, x_test_3d, x_test_f, y_ts, tr_ids, ts_n, conv_n, dense_n):
    # compute final accuracy on training and test sets
    pred_ts = model.predict([x_test_3d[:, 0], x_test_3d[:, 1], x_test_f[:, 0], x_test_f[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_ts, pred_ts)
    roc_auc = auc(fpr, tpr)
    target = open('auc_scores_summary_multi_epi.txt', 'a')
    target.write("epi, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")
    target.close()
    print("epi, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")


# create sets of 5 with 4 in training and 1 in test
def create_loo_train_test_set(data_src, data_stem_sm, data_stem_lg, train_ids, test_id):
    # get smaller patches first
    x_tr = []
    y_tr = []
    for tid in train_ids:
        train_name = data_stem_sm + str(tid)
        x_train, y_train = createShapeData.get_int_paired_format(data_src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_sm = np.concatenate(x_tr)
    y_tr_sm = np.concatenate(y_tr)

    test_name = data_stem_sm + str(test_id)
    x_test_sm, y_test_sm = createShapeData.get_int_paired_format(data_src, test_name)

    # get larger patches next
    x_tr = []
    y_tr = []
    for tid in train_ids:
        train_name = data_stem_lg + str(tid)
        x_train, y_train = createShapeData.get_int_paired_format(data_src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all_lg = np.concatenate(x_tr)
    # y_tr_all_lg = np.concatenate(y_tr)

    test_name = data_stem_lg + str(test_id)
    x_test_lg, y_test_lg = createShapeData.get_int_paired_format(data_src, test_name)

    return x_tr_sm, x_test_sm, y_tr_sm, y_test_sm, x_tr_all_lg, x_test_lg


# run this to perform cross validation
def do_cross_val(data_src, data_name_lg, data_name_sm, model_save_name):
    # data_src = src
    # data_name_lg = data_name_large
    # data_name_sm = data_name_small
    # model_save_name = save_name

    conv_n_vals = [5, 10, 15]
    dense_n_vals = [25, 50, 100]
    avail_ids = [1, 2, 3, 4, 5]
    for conv_n in conv_n_vals:
        for dense_n in dense_n_vals:
            for idi in avail_ids:
                # test on idi, train on all except idi
                test_id = idi
                tr_id = [i for i in avail_ids if i != idi]

                # get train/test pairs for validation
                x_tr_sm, x_test_sm, y_train, y_test, x_tr_lg, x_test_lg = \
                    create_loo_train_test_set(data_src, data_name_sm, data_name_lg, tr_id, test_id)

                # train and test
                model = train_model(x_tr_lg, y_train, x_tr_sm, conv_n, dense_n, model_save_name)
                run_test(model, x_test_lg, x_test_sm, y_test, tr_id, test_id, conv_n, dense_n)
                print()


# run this to get the final model
def train_final_model(data_src, data_name_lg, data_name_sm, model_save_name):
    conv_n = 15
    dense_n = 50
    tr_id = [1, 2, 3, 4, 5]
    test_id = 2
    x_tr_sm, x_test_sm, y_train, y_test, x_tr_lg, x_test_lg = \
        create_loo_train_test_set(data_src, data_name_sm, data_name_lg, tr_id, test_id)
    model = train_model(x_tr_lg, y_train, x_tr_sm, conv_n, dense_n, model_save_name)
    run_test(model, x_test_lg, x_test_sm, y_test, tr_id, test_id, conv_n, dense_n)
    print("epi, trained on: " + str(tr_id) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + "\n")


# load data
# src = '/home/nripesh/Dropbox/temp_images/run_on_allens/'
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_name_large = 'x_data_intensity_epi_large_'
data_name_small = 'x_data_intensity_epi_small_'
save_name = 'shape_match_model_epi_multi_res2.h5'

# do_cross_val(src, data_name_large, data_name_small, save_name)
train_final_model(src, data_name_large, data_name_small, save_name)

