import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Conv3D, \
    Flatten, BatchNormalization
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from sklearn.metrics import roc_curve, auc
from siamese_supervised import createShapeData
from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt


# a CNN layer for intensity inputs
def create_cnn_network(input_dim, no_conv_filt, dense_n):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    # conv layers
    seq.add(Conv3D(5, kernel_size=(3, 3, 3), input_shape=input_dim, padding='valid',
                   data_format='channels_first', activation='relu'))
    seq.add(Dropout(.2))
    seq.add(Conv3D(10, kernel_size=(3, 3, 3), padding='valid',
                   data_format='channels_first', activation='relu'))
    seq.add(Dropout(.2))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(75, activation='sigmoid'))
    return seq


# train model given x_train and y_train
def train_model(x_tr, y_tr, conv_f_n, dense_n):
    save_name = '/home/nripesh/PycharmProjects/Siamese/siamese_supervised/shape_match_model_epi_sx4.h5'
    tr_epoch = 20

    input_dim = x_tr.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = create_cnn_network(input_dim, conv_f_n, dense_n)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model_tr = Model(inputs=[input_a, input_b], outputs=distance)

    # train
    opt_func = RMSprop(lr=.003)
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    history = model_tr.fit([x_tr[:, 0], x_tr[:, 1]], y_tr, validation_split=.30,
                           batch_size=128, verbose=2, epochs=tr_epoch,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

    # summarize history for loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig('/home/nripesh/PycharmProjects/Siamese/siamese_supervised/epi_train_val_loss.png')
    plt.close(1)
    model_tr.save(save_name)
    return model_tr


# test, also provide info on which pair it was trained on and which it was tested on
def run_test(model, x_ts, y_ts, tr_ids, ts_n, conv_n, dense_n):
    # compute final accuracy on training and test sets
    pred_ts = model.predict([x_ts[:, 0], x_ts[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_ts, pred_ts)
    roc_auc = auc(fpr, tpr)
    target = open('auc_scores_summary_epi.txt', 'a')
    target.write("epi, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
                 str(roc_auc) + "\n")
    target.close()
    print("epi, trained on: " + str(tr_ids) + ", tested on: " + str(ts_n) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + ", auc: " +
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
data_stem = 'x_data_intensity_epi_'


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


# visualize images and filters - post running
def visualize():
    n_i = np.random.randint(0, x_train.shape[0])
    n_z = np.random.randint(0, 11)
    a = x_train[n_i, 0, 0, :, :, n_z]
    b = x_train[n_i, 0, 1, :, :, n_z]

    plt.figure(1)
    plt.imshow(a, interpolation='none', cmap='gray')

    plt.figure(2)
    plt.imshow(b, interpolation='none', cmap='gray')
    plt.show()

# run this to get the final model
# def train_final_model():
conv_n = 15
dense_n = 50
tr_id = [1, 3, 4, 5]
test_id = 2
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)
model = train_model(x_train, y_train, conv_n, dense_n)
run_test(model, x_test, y_test, tr_id, test_id, conv_n, dense_n)
print("epi, trained on: " + str(tr_id) + ", conv n: " + str(conv_n) + ", dense n: " + str(dense_n) + "\n")


# with dense: 0.845893070671
# no dense: .82
# with dense, all data (1, 3, 4, 5): .875
# with dense 100, all data (1, 3, 4, 5), higher learning rate: .875
# no dense, all data (1, 3, 4, 5), higher learning rate: .81
# yes dense: smaller model 5, 10 conv, dense 75: .878
# yes dense: smaller model 5, 10 conv, dense 75 (lr .003): .875
