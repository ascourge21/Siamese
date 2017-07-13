import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from face_siamese import createMNISTData

samp_f = 2
total_to_samp = 40000
total_classes = 4
x_train, y_train = createMNISTData.get_train_data_for_conv(samp_f, total_to_samp, total_classes)
x_test, y_test = createMNISTData.get_train_data_for_conv(samp_f, total_to_samp, total_classes)

inp_shape = x_train.shape[1:]

mini_batch = 8
no_epoch = 10
kern_size = 3
nb_filter = [32, 16]

model = Sequential()
# 6 filters
model.add(Convolution2D(nb_filter[0], kern_size, kern_size, input_shape=inp_shape))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
model.add(Dropout(.25))
# 12 filters
model.add(Convolution2D(nb_filter[1], kern_size, kern_size))
model.add(Activation('linear'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
model.add(Dropout(.25))

# now flatten
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('linear'))
model.add(Dropout(.25))
model.add(Dense(50))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='siamese_euclidean', optimizer=rms)

model.fit(x_train, y_train, batch_size=mini_batch, nb_epoch=no_epoch, verbose=2,
          show_accuracy=True, validation_split=.25, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

y_ts_est = model.predict(x_test, batch_size=mini_batch)
pair_errors = ((y_ts_est[0::2] - y_ts_est[1::2]) ** 2).sum(axis=1, keepdims=True)
print('replicating the loss for training: ' + str(((pair_errors - y_test[0::2])**2).mean()))


# x_train_est = model_a.predict(x_train, batch_size=mini_batch)

# plot pair errorsa
pair_error_median = np.median(pair_errors)
threshold = pair_error_median*1
plt.plot(pair_errors)
plt.hold(True)
plt.plot(threshold*np.ones(pair_errors.shape), 'r')
plt.savefig('pair_errors.png')
plt.hold(False)

label = np.zeros(pair_errors.shape)
for i in range(int(y_ts_est.shape[0]/2)):
    if pair_errors[i] > threshold:
        label[i] = 0
    else:
        label[i] = 1

y_test_norep = y_test[0::2]
y_est_norep = label
print("accuracy is: " + str(accuracy_score(y_test_norep, y_est_norep)))
print("confusion matrix:")
print(confusion_matrix(y_test_norep, y_est_norep))
