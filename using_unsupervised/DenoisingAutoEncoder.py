"""
    here we'll try denoising autoencoders. perhaps this will learn more meaningful representations than just an
    autoencoder.

    stacked denoising auto-encoder: http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf


"""


import matplotlib
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Convolution3D, \
    MaxPooling3D, UpSampling3D
from keras.layers.noise import GaussianNoise
from keras.models import Model

matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

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


def draw_patch_in_2d(im):
    f, axarr = plt.subplots(3, 4)
    for i in range(4):
        axarr[0, i].imshow(im[:, :, i * 3], interpolation='none', cmap='gray')
        axarr[0, i].set_title('xy ' + str(i * 3))

    # xz
    for i in range(4):
        axarr[1, i].imshow(im[:, i * 3, :], interpolation='none', cmap='gray')
        axarr[1, i].set_title('xz ' + str(i * 3))

    # yz
    for i in range(4):
        axarr[2, i].imshow(im[i * 3, :, :], interpolation='none', cmap='gray')
        axarr[2, i].set_title('yz ' + str(i * 3))


def visualize_results(input_im, pred_im, shp):
    draw_patch_in_2d(np.reshape(input_im, (shp[0], shp[1], shp[2])))  # intensity
    draw_patch_in_2d(np.reshape(pred_im, (shp[0], shp[1], shp[2])))  # predicted symantic labels
    plt.show()

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'x_data_intensity_comb_'

tr_id = [1, 3, 4, 5]
test_id = 2

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

x_train = np.concatenate((x_train[:, 0], x_train[:, 1]))
x_test = np.concatenate((x_test[:, 0], x_test[:, 1]))

randinds = np.random.randint(0, x_train.shape[0], x_train.shape[0])
off = 1
x_train = x_train[randinds, :, off:, off:, off:]
x_test = x_test[:, :, off:, off:, off:]

input_dim = x_train.shape[1:]
shp = list(x_train.shape)
shp = shp[2:]

# encoding layer
conv_channel_1 = 5
conv_channel_2 = 15
# conv_channel_3 = 5
kern_size = 3

input_patches = Input(shape=input_dim)

################################### ENCODER/DECODER ###################################
x_noise = GaussianNoise(sigma=.4)(input_patches)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size,
                  activation='relu', dim_ordering='th', border_mode='same')(x_noise)
x = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)
x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                  activation='relu', dim_ordering='th', border_mode='same')(x)
encoded = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)

x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(encoded)
x = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(x)
x = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x)
decoded = Convolution3D(1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                        border_mode='same')(x)
encoder = Model(input=input_patches, output=encoded)
#########################################################################################

# compile and fit model
input_w_noise = Model(input_patches, x_noise)
decoder = Model(input_patches, decoded)
decoder.compile(optimizer='adadelta', loss='mean_absolute_error')  # think about advanced losses later - like
decoder.fit(x_train, x_train,
            nb_epoch=20,
            batch_size=128,
            shuffle=True,
            verbose=2,
            validation_split=.25,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

encode_name = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_den_auto_encoder.h5'
encoder.save(encode_name)

ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
ex_1_pred = decoder.predict(ex_1)
visualize_results(ex_1, ex_1_pred, shp)

# ex_1_noise = input_w_noise.predict(ex_1)
# visualize_results(ex_1, ex_1_noise, shp)

# if encoded available, check it out
# if encoded_and_decoded:
#     ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
#     ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
#     encoded_imgs = encoder.predict(ex_1)
#     encoder.save(encode_name)

