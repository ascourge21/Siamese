import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Convolution3D, Flatten
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc

from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss
from siamese_supervised import createShapeData


# a CNN layer for intensity inputs
def create_cnn_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    nb_filter = [12, 6]
    kern_size = 3

    # conv layers
    seq.add(Convolution3D(nb_filter[0], kern_size, kern_size, kern_size, input_shape=input_dim,
                          border_mode='valid', dim_ordering='th', activation='relu'))
    # seq.add(MaxPooling3D(pool_size=(2, 2, 2)))  # downsample
    seq.add(Dropout(.25))

    # conv layer 2
    seq.add(Convolution3D(nb_filter[1], kern_size, kern_size, kern_size, border_mode='same', dim_ordering='th',
                          activation='relu'))
    # seq.add(MaxPooling3D(pool_size=(2, 2, 2), dim_ordering='th'))  # downsample
    seq.add(Dropout(.25))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(100, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

# load data
src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/matconvnet-1.0-beta21/cardiac_data/'
data_name = 'x_data_intensity_endo'
x, y = createShapeData.get_int_paired_format(src, data_name)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim = x_train.shape[2:]
input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)
base_network = create_cnn_network(input_dim)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
nb_epoch = 20
# opt_func = RMSprop(lr=.0005, clipnorm=1)
opt_func = RMSprop()
model.compile(loss=contrastive_loss, optimizer=opt_func)
model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.30,
          batch_size=100, verbose=2, nb_epoch=nb_epoch, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
model.save('shape_match_model_int_patch_endo_deep.h5')

# compute final accuracy on training and test sets
pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])


tpr, fpr, _ = roc_curve(y_test, pred_ts)
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
plt.savefig('roc_curve_endo.png')

thresh = .41
tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Mean of error less than  thresh (match): %0.3f%%' % np.mean(pred_ts[pred_ts < thresh]))
print('* Mean of error more than  thresh (no match): %0.3f%%' % np.mean(pred_ts[pred_ts >= thresh]))
print("* test case confusion matrix:")
print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test))
plt.figure(2)
plt.plot(np.concatenate([pred_ts[y_test == 1], pred_ts[y_test == 0]]))
plt.hold(True)
plt.plot(np.ones(pred_ts.shape)*thresh, 'r')
plt.hold(False)
plt.savefig('pair_errors_endo.png')
