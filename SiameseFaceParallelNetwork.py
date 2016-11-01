import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers import Input, Lambda, Dense, Dropout, merge
from keras.models import Model, Sequential
from keras.engine.topology import Merge
from keras.callbacks import EarlyStopping


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.cross_validation import train_test_split

import createFaceData
from SiameseFunctions import eucl_dist_output_shape, euclidean_distance, contrastive_loss, compute_accuracy


def create_base_network_deep(input_d):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    seq.add(Dense(200, input_shape=(input_d,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(100, activation='relu'))
    seq.add(Dropout(0.1))
    # seq.add(Dense(50, activation='relu'))
    # seq.add(Dropout(0.1))

    return seq


def create_base_network_shallow(input_d):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()

    seq.add(Dense(100, input_shape=(input_d,), activation='relu'))
    seq.add(Dropout(0.1))

    return seq

# get the data
samp_f = 3
total_to_samp = 10000
x, y = createFaceData.gen_data_new(samp_f, total_to_samp)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.30)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
input_dim = x_train.shape[2]
input_a = Input(shape=(input_dim,))
input_b = Input(shape=(input_dim,))

# stream 1
deep_network = create_base_network_deep(input_dim)
processed_a = deep_network(input_a)
processed_b = deep_network(input_b)

# stream 2
shallow_network = create_base_network_shallow(input_dim)
processed_c = shallow_network(input_a)
processed_d = shallow_network(input_b)

# merge stream 1 and 2
merged_a = merge([processed_a, processed_c], mode='concat', concat_axis=1)
merged_a = Dense(50, activation='relu')(merged_a)
merged_b = merge([processed_b, processed_d], mode='concat', concat_axis=1)
merged_b = Dense(50, activation='relu')(merged_b)

# this calls our custom distance function
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([merged_a, merged_b])

# use this for model with multiple inputs, custom output
model = Model(input=[input_a, input_b], output=distance)

# train
nb_epoch = 15
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=nb_epoch)


# compute final accuracy on training and test sets
pred_tr = model.predict([x_train[:, 0], x_train[:, 1]])
pred_ts = model.predict([x_test[:, 0], x_test[:, 1]])

# auc and other things
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
plt.savefig('roc_curve_face_parallel.png')

thresh = .35
tr_acc = accuracy_score(y_train, (pred_tr < thresh).astype('float32'))
te_acc = accuracy_score(y_test, (pred_ts < thresh).astype('float32'))
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
print('* Mean of error less than  thresh (match): %0.3f' % np.mean(pred_ts[pred_ts < thresh]))
print('* Mean of error more than  thresh (no match): %0.3f' % np.mean(pred_ts[pred_ts >= thresh]))
print("* test case confusion matrix:")
print(confusion_matrix((pred_ts < thresh).astype('float32'), y_test))
plt.figure(2)
plt.plot(np.concatenate([pred_ts[y_test == 1], pred_ts[y_test == 0]]), 'bo')
plt.hold(True)
plt.plot(np.ones(pred_ts.shape)*thresh, 'r')
plt.hold(False)
plt.savefig('pair_errors_face.png')