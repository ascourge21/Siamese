"""
    - Here the idea is to use convolutional autoencoders to see if embeddings of intensity patches
    can be found
    - the best model (the encoding part) is also likely to work better in the supervised context
    - also transfer learning can be done to train in the supervised context or train on real data.
    - specially with real data - there's tonnes and tonnes of unlabeld patches so need to worry.

    - First try simple 1 or 2 dense layer model - baseline
    - Then try convolution + deeper models
"""

import matplotlib
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Convolution3D, \
    MaxPooling3D, UpSampling3D
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


def visualize_results(input_im, pred_im):
    im1 = np.reshape(input_im, (input_im.shape[2], input_im.shape[3], input_im.shape[4]))
    f, axarr = plt.subplots(3, 4)
    # xy
    axarr[0, 0].imshow(im1[:, :, 0], interpolation='none', cmap='gray')
    axarr[0, 0].set_title('xy 0')
    axarr[0, 1].imshow(im1[:, :, 3], interpolation='none', cmap='gray')
    axarr[0, 1].set_title('xy 3')
    axarr[0, 2].imshow(im1[:, :, 5], interpolation='none', cmap='gray')
    axarr[0, 2].set_title('xy 5')
    axarr[0, 3].imshow(im1[:, :, 7], interpolation='none', cmap='gray')
    axarr[0, 3].set_title('xy 7')
    #xz
    axarr[1, 0].imshow(im1[:, 0, :], interpolation='none', cmap='gray')
    axarr[1, 0].set_title('xz 0')
    axarr[1, 1].imshow(im1[:, 3, :], interpolation='none', cmap='gray')
    axarr[1, 1].set_title('xz 5')
    axarr[1, 2].imshow(im1[:, 5, :], interpolation='none', cmap='gray')
    axarr[1, 2].set_title('xz 3')
    axarr[1, 3].imshow(im1[:, 7, :], interpolation='none', cmap='gray')
    axarr[1, 3].set_title('xz 7')
    #yz
    axarr[2, 0].imshow(im1[0, :, :], interpolation='none', cmap='gray')
    axarr[2, 0].set_title('yz 0')
    axarr[2, 1].imshow(im1[3, :, :], interpolation='none', cmap='gray')
    axarr[2, 1].set_title('yz 5')
    axarr[2, 2].imshow(im1[5, :, :], interpolation='none', cmap='gray')
    axarr[2, 2].set_title('yz 3')
    axarr[2, 3].imshow(im1[7, :, :], interpolation='none', cmap='gray')
    axarr[2, 3].set_title('yz 7')

    im2 = np.reshape(pred_im, (pred_im.shape[2], pred_im.shape[3], pred_im.shape[4]))
    f, axarr2 = plt.subplots(3, 4)
    # xy
    axarr2[0, 0].imshow(im2[:, :, 0], interpolation='none', cmap='gray')
    axarr2[0, 0].set_title('xy 0')
    axarr2[0, 1].imshow(im2[:, :, 3], interpolation='none', cmap='gray')
    axarr2[0, 1].set_title('xy 3')
    axarr2[0, 2].imshow(im2[:, :, 5], interpolation='none', cmap='gray')
    axarr2[0, 2].set_title('xy 5')
    axarr2[0, 3].imshow(im2[:, :, 7], interpolation='none', cmap='gray')
    axarr2[0, 3].set_title('xy 7')
    #xz
    axarr2[1, 0].imshow(im2[:, 0, :], interpolation='none', cmap='gray')
    axarr2[1, 0].set_title('xz 0')
    axarr2[1, 1].imshow(im2[:, 3, :], interpolation='none', cmap='gray')
    axarr2[1, 1].set_title('xz 5')
    axarr2[1, 2].imshow(im2[:, 5, :], interpolation='none', cmap='gray')
    axarr2[1, 2].set_title('xz 3')
    axarr2[1, 3].imshow(im2[:, 7, :], interpolation='none', cmap='gray')
    axarr2[1, 3].set_title('xz 7')
    #yz
    axarr2[2, 0].imshow(im2[0, :, :], interpolation='none', cmap='gray')
    axarr2[2, 0].set_title('yz 0')
    axarr2[2, 1].imshow(im2[3, :, :], interpolation='none', cmap='gray')
    axarr2[2, 1].set_title('yz 5')
    axarr2[2, 2].imshow(im2[5, :, :], interpolation='none', cmap='gray')
    axarr2[2, 2].set_title('yz 3')
    axarr2[2, 3].imshow(im2[7, :, :], interpolation='none', cmap='gray')
    axarr2[2, 3].set_title('yz 7')

    plt.show()

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'x_data_intensity_comb_'

tr_id = [1, 3, 4, 5]
test_id = 2

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

x_train = np.concatenate((x_train[:, 0], x_train[:, 1]))
x_test = np.concatenate((x_test[:, 0], x_test[:, 1]))

# randinds = np.random.randint(0, x_train.shape[0], x_train.shape[0])
off = 1
# x_train = x_train[randinds, :, off:, off:, off:]
x_train = x_train[:, :, off:, off:, off:]
x_test = x_test[:, :, off:, off:, off:]

input_dim = x_train.shape[1:]

# encoding layer
conv_channel_1 = 5
conv_channel_2 = 15
# conv_channel_3 = 5
kern_size = 3

input_patches = Input(shape=input_dim)

################################### ENCODER/DECODER ###################################
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                  activation='relu', dim_ordering='th', border_mode='same')(input_patches)
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
decoder = Model(input_patches, decoded)
decoder.compile(optimizer='adadelta', loss='mean_absolute_error') # think about advanced losses later - like
decoder.fit(x_train, x_train,
            nb_epoch=20,
            batch_size=128,
            shuffle=True,
            verbose=2,
            validation_data=(x_test, x_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

encode_name = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_int_encoder.h5'
encoder.save(encode_name)

ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
ex_1_pred = decoder.predict(ex_1)
visualize_results(ex_1, ex_1_pred)


# if encoded available, check it out
# if encoded_and_decoded:
#     ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
#     ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
#     encoded_imgs = encoder.predict(ex_1)
#     encoder.save(encode_name)

