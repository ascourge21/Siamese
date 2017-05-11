import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout, Convolution3D, \
    MaxPooling3D, Flatten, BatchNormalization, merge
from keras.regularizers import WeightRegularizer, l2
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import train_test_split

import createShapeData
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss


# a CNN layer for intensity inputs
def create_cnn_network(input_dim, no_conv_filt, dense_n):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    # conv layers
    kern_size = 3
    seq.add(Convolution3D(5, kern_size, kern_size, kern_size, input_shape=input_dim, subsample=(2, 2, 2),
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    kern_size = 3
    seq.add(Convolution3D(15, kern_size, kern_size, kern_size,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(dense_n, activation='relu'))
    seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))
    return seq


# a network with a couple dense layers
def create_simple_network(input_dim):
    '''Dense Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(100, input_shape=(input_dim,), activation='relu'))
    seq.add(BatchNormalization(mode=2))
    seq.add(Dropout(0.2))
    seq.add(Dense(50, activation='relu'))
    seq.add(BatchNormalization(mode=2))
    return seq


# train model given x_train and y_train
def train_model(x_train_int, x_train_sh, y_train, conv_n, dense_n):
    save_name = 'int_shctxt_match_model_endo.h5'
    tr_epoch = 10

    # the convolutional part
    input_dim = x_train_int.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    cnn_network = create_cnn_network(input_dim, conv_n, dense_n)
    processed_a = cnn_network(input_a)
    processed_b = cnn_network(input_b)

    # the fully connected shape context part
    input_dim2 = x_train_sh.shape[2]
    input_c = Input(shape=(input_dim2,))
    input_d = Input(shape=(input_dim2,))
    fc_network = create_simple_network(input_dim2)
    processed_c = fc_network(input_c)
    processed_d = fc_network(input_d)

    # merge
    merged_a = merge([processed_a, processed_c], mode='concat', concat_axis=1)
    merged_b = merge([processed_b, processed_d], mode='concat', concat_axis=1)

    # add a fc layer after merge
    # merged_a = Dense(25, activation='relu')(merged_a)
    # merged_b = Dense(25, activation='relu')(merged_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([merged_a, merged_b])

    model_tr = Model(input=[input_a, input_b, input_c, input_d], output=distance)

    # train
    # opt_func = RMSprop(lr=.0005, clipnorm=1)
    opt_func = RMSprop()
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    model_tr.fit([x_train_int[:, 0], x_train_int[:, 1], x_train_sh[:, 0], x_train_sh[:, 1]],
                 y_train, validation_split=.30, batch_size=128, verbose=2, nb_epoch=tr_epoch,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model_tr.save(save_name)
    return model_tr


# test, also provide info on which pair it was trained on and which it was tested on
def run_test(model, x_test_int, x_test_shp, y_ts, tr_ids, ts_n, conv_n, dense_n):
    # compute final accuracy on training and test sets
    pred_ts = model.predict([x_test_int[:, 0], x_test_int[:, 1], x_test_shp[:, 0], x_test_shp[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_ts, pred_ts)
    roc_auc = auc(fpr, tpr)

    print("endo, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")


# create groups of 4 image sets as training and 1 as test
def create_loo_train_test_set(src, data_stem_int, data_stem_shp, train_id, test_id):
    # first get the intensity data
    x_tr = []
    y_tr = []
    for tid in train_id:
        train_name = data_stem_int + str(tid)
        x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all = np.concatenate(x_tr)
    y_tr_all = np.concatenate(y_tr)

    test_name = data_stem_int + str(test_id)
    x_test, y_test = createShapeData.get_int_paired_format(src, test_name)

    # now get the shape data
    x_tr_sh = []
    for tid in train_id:
        train_name = data_stem_shp + str(tid)
        x, _ = createShapeData.get_shctxt_paired_format(src, train_name)
        x_tr_sh.append(x)

    x_tr_all_sh = np.concatenate(x_tr_sh)

    test_name = data_stem_shp + str(test_id)
    x_test_sh, _ = createShapeData.get_shctxt_paired_format(src, test_name)

    return x_tr_all, x_test, y_tr_all, y_test, x_tr_all_sh, x_test_sh


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

                x_tr_all, x_test, y_tr_all, y_test, x_tr_all_sh, x_test_sh = \
                    create_loo_train_test_set(src, data_stem_int, data_stem_shp, tr_id, test_id)
                model = train_model(x_tr_all, x_tr_all_sh, y_train, conv_n, dense_n)
                run_test(model, x_test, x_test_sh, y_test, tr_id, test_id, conv_n, dense_n)
                print()


# visualize images and filters - post running
def visualize():
    n_i = np.random.randint(0, x_train_int.shape[0])
    n_z = np.random.randint(0, x_train_int.shape[3])
    a = x_train_int[n_i, 0, 0, :, :, n_z]
    b = x_train_int[n_i, 1, 0, :, :, n_z]

    print("n_i:" + str(n_i) + ", y: " + str(y_train[n_i]))

    plt.figure(1)
    plt.imshow(a, interpolation='none', cmap='gray')

    plt.figure(2)
    plt.imshow(b, interpolation='none', cmap='gray')

    a = x_train_sh[n_i, 0, :]
    b = x_train_sh[n_i, 1, :]

    plt.figure(3)
    plt.plot(a)

    plt.figure(4)
    plt.plot(b)

    plt.show()


# load 1 and 2 and test on 3
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
# src = '/home/nripesh/Dropbox/temp_images/run_on_allens/'
data_stem_int = 'x_data_intensity_endo_'
data_stem_shp = 'x_data_shctxt_endo_'


# run this to get the final model
# def train_final_model():
conv_n = 15
dense_n = 100
tr_id = [1, 3, 4, 5]  # too large with all of them
test_id = 2
x_train_int, x_test_int, y_train, y_test, x_train_sh, x_test_sh = \
    create_loo_train_test_set(src, data_stem_int, data_stem_shp, tr_id, test_id)

model = train_model(x_train_int, x_train_sh, y_train, conv_n, dense_n)
run_test(model, x_test_int, x_test_sh, y_test, tr_id, test_id, conv_n, dense_n)
print("endo, trained on: " + str(tr_id) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + "\n")


# do_cross_val()
# train_final_model()