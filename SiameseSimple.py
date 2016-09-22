import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Dropout, Activation


# 100 data points, 10 dimension - a and b are matching
X_matching_a = np.random.rand(100, 40)
X_matching_b = np.random.rand(100, 40)

# 200 data points, 10 dimension - a and b are non matching
X_non_matching_a = np.random.rand(200, 40)
X_non_matching_b = np.random.rand(200, 40)


def arrange_input_and_outputs(X_m_a, X_m_b, X_n_a, X_n_b):
    X_comb_m = np.zeros([X_m_a.shape[0]*2, X_m_a.shape[1]])
    Y_comb_m = np.zeros([X_m_a.shape[0]*2, ])

    count = 0
    for i in range(X_m_a.shape[0]):
        X_comb_m[count, :] = X_m_a[i, :]
        Y_comb_m[count] = 1
        count += 1
        X_comb_m[count, :] = X_m_b[i, :]
        Y_comb_m[count] = 1
        count += 1

    X_comb_n = np.zeros([X_n_a.shape[0] * 2, X_n_a.shape[1]])
    Y_comb_n = np.zeros([X_n_a.shape[0] * 2, ])

    count = 0
    for i in range(X_n_a.shape[0]):
        X_comb_n[count, :] = X_n_a[i, :]
        Y_comb_n[count] = 0
        count += 1
        X_comb_n[count, :] = X_n_b[i, :]
        Y_comb_n[count] = 0
        count += 1

    X_comb = np.concatenate([X_comb_m, X_comb_n], axis=0)
    Y_comb = np.concatenate([Y_comb_m, Y_comb_n], axis=0)

    return X_comb, Y_comb

X_train, Y_train = arrange_input_and_outputs(X_matching_a, X_matching_b, X_non_matching_a, X_non_matching_b)

model_a = Sequential()
model_a.add(Dense(20, input_shape=(40,)))
model_a.add(Activation('relu'))
model_a.add(Dense(10))
model_a.add(Activation('relu'))

rms = RMSprop()
model_a.compile(loss='siamese_euclidean', optimizer=rms)

mini_batch = 32
no_epoch = 10
model_a.fit(X_train, Y_train, batch_size=mini_batch, nb_epoch=no_epoch, verbose=1,
          show_accuracy=True, validation_split=.25)

#  callbacks=[EarlyStopping(monitor='val_loss', patience=2)]
# y_ts_est = model_a.predict(X_test, batch_size=mini_batch)
# y_ts_l = np.argmax(Y_test, 1)
# y_ts_l_est = np.argmax(y_ts_est, 1)
# print("accuracy is: " + str(accuracy_score(y_ts_l, y_ts_l_est)))
# print("confusion matrix:")
# print(confusion_matrix(y_ts_l, y_ts_l_est))




