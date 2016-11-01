import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout, Convolution3D, MaxPooling3D, Flatten, merge
from keras.regularizers import WeightRegularizer, l2
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import train_test_split

import createShapeData
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss


# a CNN layer for intensity inputs
def create_cnn_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layers
    seq.add(Convolution3D(12, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    # seq.add(MaxPooling3D(pool_size=(2, 2, 2)))  # downsample
    seq.add(Dropout(.1))

    # dense layer
    seq.add(Flatten())
    seq.add(Dense(100, activation='relu'))
    return seq


# a CNN layer for intensity inputs
def create_cnn_network_small(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    kern_size = 3

    # conv layers
    seq.add(Convolution3D(8, kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    # seq.add(MaxPooling3D(pool_size=(2, 2, 2)))  # downsample
    seq.add(Dropout(.1))

    # dense layer
    seq.add(Flatten())
    seq.add(Dense(50, activation='relu'))
    return seq


# load data
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/matconvnet-1.0-beta21/cardiac_data/'
data_name_large = 'x_data_intensity_epi_large'
data_name_small = 'x_data_intensity_epi_small'
save_name = 'shape_match_model_epi_multi_res2.h5'

# the larger patches
x3d, y3d = createShapeData.get_int_paired_format(src, data_name_large)
x_train_3d, x_test_3d, y_train_3d, y_test_3d = train_test_split(x3d, y3d, test_size=.25)

# the smaller patches
xf, yf = createShapeData.get_int_paired_format(src, data_name_small)
x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(xf, yf, test_size=.25)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim_lg = x_train_3d.shape[2:]
input_a = Input(shape=input_dim_lg)
input_b = Input(shape=input_dim_lg)

input_dim_sm = x_train_f.shape[2:]
input_c = Input(shape=input_dim_sm)
input_d = Input(shape=input_dim_sm)

# the layer with convolutions
cnn_network = create_cnn_network(input_dim_lg)
processed_a = cnn_network(input_a)
processed_b = cnn_network(input_b)

# the layer without convolutions, smaller patches
dense_network = create_cnn_network_small(input_dim_sm)
processed_c = dense_network(input_c)
processed_d = dense_network(input_d)

# merge dense and cnn - and add a dense layer on top of  that
merged_a = merge([processed_a, processed_c], mode='concat', concat_axis=1)
merged_a = Dense(50, activation='relu')(merged_a)
merged_b = merge([processed_b, processed_d], mode='concat', concat_axis=1)
merged_b = Dense(50, activation='relu')(merged_b)

# custom distance
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([merged_a, merged_b])

# the model, finally
model = Model(input=[input_a, input_b, input_c, input_d], output=distance)

# train
nb_epoch = 15
opt_func = RMSprop()
model.compile(loss=contrastive_loss, optimizer=opt_func)
model.fit([x_train_3d[:, 0], x_train_3d[:, 1], x_train_f[:, 0], x_train_f[:, 1]], y_train_3d, validation_split=.30,
          batch_size=32, verbose=2, nb_epoch=nb_epoch, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
model.save(save_name)

# compute final accuracy on training and test sets
pred_tr = model.predict([x_train_3d[:, 0], x_train_3d[:, 1], x_train_f[:, 0], x_train_f[:, 1]])
pred_ts = model.predict([x_test_3d[:, 0], x_test_3d[:, 1], x_test_f[:, 0], x_test_f[:, 1]])


tpr, fpr, _ = roc_curve(y_test_3d, pred_ts)
roc_auc = auc(fpr, tpr)

plt.figure(1)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.hold(True)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.hold(False)
plt.savefig('roc_curve_epi_multires_conv.png')

thresh = .41
tr_acc = accuracy_score(y_train_3d, (pred_tr < thresh).astype('float32'))
te_acc = accuracy_score(y_test_3d, (pred_ts < thresh).astype('float32'))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Mean of error less than  thresh (match): %0.3f' % np.mean(pred_ts[pred_ts < thresh]))
print('* Mean of error more than  thresh (no match): %0.3f' % np.mean(pred_ts[pred_ts >= thresh]))
print("* test case confusion matrix:")
print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test_3d))
plt.figure(2)
plt.plot(np.concatenate([pred_ts[y_test_3d == 1], pred_ts[y_test_3d == 0]]), 'bo')
plt.hold(True)
plt.plot(np.ones(pred_ts.shape)*thresh, 'r')
plt.hold(False)
plt.savefig('pair_errors_epi_multires_conv.png')
