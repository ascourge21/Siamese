from keras.layers import Flatten,Dropout, Dense,Activation,Convolution2D,MaxPooling2D, Reshape
from keras.models import Sequential
import numpy as np

data=np.array([[ 0.06598175,  0.12817432,  0.36949296,  0.5712227 ,
         0.54105666, 0.38620151,  0.20075556,  0.11366828,  0.07925526,
         0.05270926, 0.08046888,  0.07579738],
       [ 0.06797359,  0.13533782,  0.37340452,  0.57014065,  0.53585215,
        0.39121396,  0.20627117,  0.10934431,  0.07212514,  0.04386237,
        0.07302206,  0.07364738]])

dataT = data.reshape(data.shape[0], 1, 1, 12)

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(5, 1, 3, border_mode='valid', input_shape=(1, 1, 12), dim_ordering='th'))
model.add(Convolution2D(12, 1, 3, dim_ordering='th'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), dim_ordering='th'))
# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(48))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(12))
model.add(Activation('softmax'))
model.add(Reshape((1, 1, 12)))

model.compile(loss='mse', optimizer='SGD')
model.fit(dataT, dataT, batch_size=2, nb_epoch=1)