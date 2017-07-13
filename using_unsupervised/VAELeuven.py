'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import matplotlib
import numpy as np

matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

from keras.layers import Input, Convolution3D, \
    MaxPooling3D, UpSampling3D

from siamese_supervised import createShapeData


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

########################################################
######################  train test data

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'x_data_intensity_comb_'
tr_id = [1]
test_id = 2

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

x_train = np.concatenate((x_train[:, 0], x_train[:, 1]))
x_test = np.concatenate((x_test[:, 0], x_test[:, 1]))

# randinds = np.random.randint(0, x_train.shape[0], x_train.shape[0])
off = 1
x_train = x_train[0:1000, :, off:5, off:5, off:5]
x_test = x_test[0:1000, :, off:5, off:5, off:5]

input_dim = x_train.shape[1:]

#################  encoder  #########################
batch_size = 16
epsilon_std = 0.01
nb_epoch = 20
latent_dim = (1, 2, 2, 2)
x = Input(batch_shape=(batch_size, input_dim[0], input_dim[1], input_dim[2], input_dim[3]))
conv_channel_1 = 1
kern_size = 2
h = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                  activation='relu', dim_ordering='th', border_mode='same')(x)
z_mean = MaxPooling3D((2, 2, 2), dim_ordering='th')(h)
z_log_sigma = MaxPooling3D((2, 2, 2), dim_ordering='th')(h)
#####################################################


def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim[0], latent_dim[1], latent_dim[2], latent_dim[3]),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

# we instantiate these layers separately so as to reuse them later
decoder_h = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='sigmoid', dim_ordering='th',
                          border_mode='same')
decoder_mean = UpSampling3D(size=(2, 2, 2), dim_ordering='th')
# decoder_h = Dense(intermediate_dim, activation='relu')
# decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)




vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        verbose=2,
        validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()