"""
    Here we'll attempt at transferring the simulated model's weights and train top layers
    with the cardiac data.

    # to run directly on terminal
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python3 TransferSimModelToDsea.py
"""

import numpy as np
from keras.optimizers import RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout, Convolution3D, MaxPooling3D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import train_test_split
import createShapeData
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss
from keras.models import load_model


# a CNN layer for intensity inputs
def create_cnn_network(input_dim, no_conv_filt):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layers
    seq.add(Convolution3D(no_conv_filt, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    # seq.add(MaxPooling3D(pool_size=(2, 2, 2)))  # downsample
    seq.add(Dropout(.1))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(100, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

# check if training from scratch or loading a used model
train_from_scratch = False

if train_from_scratch:
    # load Leuven data
    src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
    data_name = 'x_data_intensity_endo_epi.mat'
    save_name = 'leuven_model_to_transfer.h5'
    x, y = createShapeData.get_int_paired_format(src, data_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)

    # because we re-use the same instance `base_network`,
    # the weights of the network will be shared across the two branches
    nb_epoch = 20
    conv_f_n = 15
    input_dim = x_train.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = create_cnn_network(input_dim, conv_f_n)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)

    # train and save
    opt_func = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=opt_func)
    model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
              batch_size=32, verbose=2, nb_epoch=nb_epoch, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
    model.save(save_name)

    # compute final accuracy on test sets
    pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])

    # get auc scores
    tpr, fpr, _ = roc_curve(y_test, pred_ts)
    roc_auc = auc(fpr, tpr)
    print("roc auc for training on the simulated dataset: " + str(roc_auc))
else:
    model = load_model('/home/nripesh/PycharmProjects/Siamese/leuven_model_to_transfer.h5')

"""
    NOW APPLY TO DSEA DATA
"""
# first freeze if necessary
to_freeze = True
if to_freeze:
    # freeze the feature generation layers - no need to train these.
    N_TO_FREEZE = 2
    for i in range(len(model.layers[2].layers)):
        if i < N_TO_FREEZE:
            model.layers[2].layers[i].Trainable = False
            print('frozen: ' + str(model.layers[2].layers[i]))

# load data
dsea_src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/' \
      'dsea_data_based_train_patches/'
data_dsea_name = 'dsea_data_patch_pairs_size_7'
save_dsea_name = 'dsea_trf_match_model.h5'

x_d, y_d = createShapeData.get_int_paired_format(dsea_src, data_dsea_name)
x_train_d, x_test_d, y_train_d, y_test_d = train_test_split(x_d, y_d, test_size=.25)


# compile and train again
nb_epoch_dsea = 25
opt_func = RMSprop()
model.compile(loss=contrastive_loss, optimizer=opt_func)
model.fit([x_train_d[:, 0], x_train_d[:, 1]], y_train_d, validation_split=.25,
          batch_size=32, verbose=2, nb_epoch=nb_epoch_dsea, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
model.save('/home/nripesh/PycharmProjects/Siamese/real_data/' + save_dsea_name)

# compute final accuracy on training and test sets
pred_tr = model.predict([x_train_d[:, 0], x_train_d[:, 1]])
pred_ts = model.predict([x_test_d[:, 0], x_test_d[:, 1]])

# get auc scores
tpr, fpr, _ = roc_curve(y_test_d, pred_ts)
roc_auc = auc(fpr, tpr)
print('AUC score: ' + str(roc_auc))

plt.figure(1)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.hold(True)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC example')
plt.legend(loc="lower right")
plt.hold(False)
plt.savefig('roc_curve_trans_dsea.png')

thresh = .5
tr_acc = accuracy_score(y_train_d, (pred_tr < thresh).astype('float32'))
te_acc = accuracy_score(y_test_d, (pred_ts < thresh).astype('float32'))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Mean of error less than  thresh (match): %0.3f%%' % np.mean(pred_ts[pred_ts < thresh]))
print('* Mean of error more than  thresh (no match): %0.3f%%' % np.mean(pred_ts[pred_ts >= thresh]))
print("* test case confusion matrix:")
print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test_d))
plt.figure(2)
plt.plot(np.concatenate([pred_ts[y_test_d == 1], pred_ts[y_test_d == 0]]), 'bo')
plt.hold(True)
plt.plot(np.ones(pred_ts.shape)*thresh, 'r')
plt.hold(False)
plt.savefig('pair_errors_dsea_trans.png')
