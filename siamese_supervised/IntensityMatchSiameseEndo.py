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
def create_cnn_network(input_dim, no_conv_filt):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    nb_filter = [no_conv_filt, 6]
    kern_size = 3

    # conv layers
    seq.add(Convolution3D(nb_filter[0], kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    seq.add(Dropout(.1))
    seq.add(BatchNormalization(mode=2))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(50, activation='relu'))
    seq.add(BatchNormalization(mode=2))
    # seq.add(Dense(50, activation='relu'))
    # seq.add(BatchNormalization(mode=2))
    return seq


# train model given x_train and y_train
def train_model(x_tr, y_tr, conv_f_n):
    save_name = 'shape_match_model_endo.h5'
    tr_epoch = 15

    input_dim = x_tr.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = create_cnn_network(input_dim, conv_f_n)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model_tr = Model(input=[input_a, input_b], output=distance)

    # train
    # opt_func = RMSprop(lr=.0005, clipnorm=1)
    opt_func = RMSprop()
    model_tr.compile(loss=contrastive_loss, optimizer=opt_func)
    model_tr.fit([x_tr[:, 0], x_tr[:, 1]], y_tr, validation_split=.25,
              batch_size=32, verbose=2, nb_epoch=tr_epoch, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model_tr.save(save_name)
    return model_tr


# test, also provide info on which pair it was trained on and which it was tested on
def run_test(model, x_ts, y_ts, tr1_n, tr2_n, ts_n):
    # compute final accuracy on training and test sets
    pred_ts = model.predict([x_ts[:, 0], x_ts[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_ts, pred_ts)
    roc_auc = auc(fpr, tpr)
    target = open('auc_scores_summary.txt', 'a')
    target.write("endo, trained on: (" + str(tr1_n) + ", " + str(tr2_n) + ") , tested on: " + str(ts_n) + ", auc: " +
                 str(roc_auc) + "\n")
    target.close()
    print("endo, trained on: (" + str(tr1_n) + ", " + str(tr2_n) + ") , tested on: " + str(ts_n) + ", auc: " +
                 str(roc_auc) + "\n")


# load 1 and 2 and test on 3
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
# train_name = 'x_data_intensity_endo_1_2'
# test_name = 'x_data_intensity_endo_3'
# x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
# x_test, y_test = createShapeData.get_int_paired_format(src, test_name)
# model = train_model(x_train, y_train, 12)
# run_test(model, x_test, y_test, 1, 2, 3)
#
# # load 1 and 3 and test on 2
# train_name = 'x_data_intensity_endo_1_3'
# test_name = 'x_data_intensity_endo_2'
# x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
# x_test, y_test = createShapeData.get_int_paired_format(src, test_name)
# model = train_model(x_train, y_train, 12)
# run_test(model, x_test, y_test, 1, 3, 2)

# load 2 and 3 and test on 1
train_name = 'x_data_intensity_endo_2_3'
test_name = 'x_data_intensity_endo_1'
x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
x_test, y_test = createShapeData.get_int_paired_format(src, test_name)
model = train_model(x_train, y_train, 12)
run_test(model, x_test, y_test, 2, 3, 1)

# final model, train on all group
train_name = 'x_data_intensity_endo_all'
x_train, y_train = createShapeData.get_int_paired_format(src, train_name)
x_test, y_test = createShapeData.get_int_paired_format(src, test_name)
model = train_model(x_train, y_train, 12)
print("endo trained on: all data")
# run_test(model, x_test, y_test, 2, 3, 1)
