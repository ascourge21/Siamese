"""
    Here, we'll learn the relationship between the
        input - image patches and the
        output - semantic segmentation labeling.
    We'll use the same code as the auto-encoding, but instead of reconstructing the input,
        we'll reconstruct the segmentation.
"""

import matplotlib
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Convolution3D, \
    MaxPooling3D, UpSampling3D
from keras.models import Model, Sequential

matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

from siamese_supervised import createShapeData


# create groups of 4 image sets as training and 1 as test
def create_loo_train_test_set(src, data_stem, train_ids, test_id):
    x_tr = []
    y_tr = []
    for tid in train_ids:
        train_name = data_stem + str(tid)
        x_train, y_train = createShapeData.get_patches_and_symantic_labels(src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all = np.concatenate(x_tr)
    y_tr_all = np.concatenate(y_tr)

    test_name = data_stem + str(test_id)
    x_test, y_test = createShapeData.get_patches_and_symantic_labels(src, test_name)
    return x_tr_all, x_test, y_tr_all, y_test


def visualize_results(input_im, input_label, pred_im):
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

    ################### labels
    im_lab = np.reshape(input_label, (input_label.shape[1], input_label.shape[2], input_label.shape[3]))
    f, axarr = plt.subplots(3, 4)
    # xy
    axarr[0, 0].imshow(im_lab[:, :, 0], interpolation='none', cmap='gray')
    axarr[0, 0].set_title('xy 0')
    axarr[0, 1].imshow(im_lab[:, :, 3], interpolation='none', cmap='gray')
    axarr[0, 1].set_title('xy 3')
    axarr[0, 2].imshow(im_lab[:, :, 5], interpolation='none', cmap='gray')
    axarr[0, 2].set_title('xy 5')
    axarr[0, 3].imshow(im_lab[:, :, 7], interpolation='none', cmap='gray')
    axarr[0, 3].set_title('xy 7')
    #xz
    axarr[1, 0].imshow(im_lab[:, 0, :], interpolation='none', cmap='gray')
    axarr[1, 0].set_title('xz 0')
    axarr[1, 1].imshow(im_lab[:, 3, :], interpolation='none', cmap='gray')
    axarr[1, 1].set_title('xz 5')
    axarr[1, 2].imshow(im_lab[:, 5, :], interpolation='none', cmap='gray')
    axarr[1, 2].set_title('xz 3')
    axarr[1, 3].imshow(im_lab[:, 7, :], interpolation='none', cmap='gray')
    axarr[1, 3].set_title('xz 7')
    #yz
    axarr[2, 0].imshow(im_lab[0, :, :], interpolation='none', cmap='gray')
    axarr[2, 0].set_title('yz 0')
    axarr[2, 1].imshow(im_lab[3, :, :], interpolation='none', cmap='gray')
    axarr[2, 1].set_title('yz 5')
    axarr[2, 2].imshow(im_lab[5, :, :], interpolation='none', cmap='gray')
    axarr[2, 2].set_title('yz 3')
    axarr[2, 3].imshow(im_lab[7, :, :], interpolation='none', cmap='gray')
    axarr[2, 3].set_title('yz 7')

    ################ predicted symantic labels
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

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/' \
      'dsea_data_based_train_patches/'
data_stem = 'dsea_labeled_semantic_patches_'

tr_id = [19, 22, 23, 61, 65, 74, 75]
test_id = 71
N_epoch = 40

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

# randinds = np.random.randint(0, x_train.shape[0], x_train.shape[0])
off = 1
# x_train = x_train[randinds, :, off:, off:, off:]
x_train = x_train[:, :, off:, off:, off:]
x_test = x_test[:, :, off:, off:, off:]
y_train = y_train[:, :, off:, off:, off:]
y_test = y_test[:, :, off:, off:, off:]

input_dim = x_train.shape[1:]

# encoding layer
conv_channel_1 = 5
conv_channel_2 = 15
kern_size = 3

input_patches = Input(shape=input_dim)

encoded_and_decoded = True
if encoded_and_decoded:
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
else:
    decoded = Sequential()
    # encode
    decoded.add(Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                              activation='relu', dim_ordering='th', border_mode='same'))
    decoded.add(MaxPooling3D((2, 2, 2), dim_ordering='th'))
    decoded.add(Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                              activation='relu', dim_ordering='th', border_mode='same'))
    decoded.add(MaxPooling3D((2, 2, 2), dim_ordering='th'))
    # decode
    decoded.add(Convolution3D(conv_channel_2, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                              border_mode='same'))
    decoded.add(UpSampling3D(size=(2, 2, 2), dim_ordering='th'))
    decoded.add(Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                              border_mode='same'))
    decoded.add(UpSampling3D(size=(2, 2, 2), dim_ordering='th'))
    decoded.add(Convolution3D(1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                              border_mode='same'))
    decoded = decoded(input_patches)

# compile and fit model
decoder = Model(input_patches, decoded)
decoder.compile(optimizer='adadelta', loss='mean_absolute_error')
decoder.fit(x_train, y_train,
            nb_epoch=N_epoch,
            batch_size=128,
            shuffle=True,
            verbose=2,
            validation_split=.25,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

encode_name = '/home/nripesh/PycharmProjects/Siamese/real_data/dsea_semantic_encoder.h5'
encoder.save(encode_name)

rand_int1 = np.random.randint(0, x_test.shape[0])
ex_1 = x_test[rand_int1, :]
ex_1_label = y_test[rand_int1, :]
ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
ex_1_pred = decoder.predict(ex_1)
visualize_results(ex_1, ex_1_label, ex_1_pred)


# # if encoded available, check it out
# if encoded_and_decoded:
#     ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
#     ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
#     encoded_imgs = encoder.predict(ex_1)
#     encoder.save(encode_name)

