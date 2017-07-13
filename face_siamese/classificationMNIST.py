import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from face_siamese import createMNISTData

x, y = createMNISTData.get_data_for_classification()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=.25)

inp_shape = x_train.shape[1:]

mini_batch = 32
no_epoch = 10
kern_size = 3
nb_filter = [6, 16]

model = Sequential()
# 6 filters
model.add(Convolution2D(nb_filter[0], kern_size, kern_size, input_shape=inp_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
model.add(Dropout(.25))
# 12 filters
model.add(Convolution2D(nb_filter[1], kern_size, kern_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
model.add(Dropout(.25))

# now flatten
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Dense(10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(x_train, y_train, batch_size=mini_batch, nb_epoch=no_epoch, verbose=1,
          show_accuracy=True, validation_split=.25, callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

y_ts_est = model.predict(x_test, batch_size=mini_batch)
y_ts = np.argmax(y_ts_est, axis=1)
y_ts_true = np.argmax(y_test, axis=1)

print("accuracy is: " + str(accuracy_score(y_ts, y_ts_true)))
print("confusion matrix:")
print(confusion_matrix(y_ts, y_ts_true))