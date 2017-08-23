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
from keras.layers import Input, Conv3D, \
    MaxPooling3D, UpSampling3D
from keras.layers.merge import Concatenate
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
        x_train, y_train = createShapeData.get_patches_and_symantic_labels(src, train_name)
        x_tr.append(x_train)
        y_tr.append(y_train)

    x_tr_all = np.concatenate(x_tr)
    y_tr_all = np.concatenate(y_tr)

    test_name = data_stem + str(test_id)
    x_test, y_test = createShapeData.get_patches_and_symantic_labels(src, test_name)
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


def visualize_results(input_im, input_label, pred_im, shp):
    draw_patch_in_2d(np.reshape(input_im, (shp[0], shp[1], shp[2])))  # intensity
    draw_patch_in_2d(np.reshape(input_label, (shp[0], shp[1], shp[2])))  # segmentation
    draw_patch_in_2d(np.reshape(pred_im, (shp[0], shp[1], shp[2])))  # predicted symantic labels
    plt.show()


src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'leuven_labeled_semantic_patches_'

tr_id = [25, 27, 28, 29]
# tr_id = [25]
test_id = 26

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

# randinds = np.random.randint(0, x_train.shape[0], x_train.shape[0])
off = 1
# x_train = x_train[randinds, :, off:, off:, off:]
x_train = x_train[:, :, off:, off:, off:]
x_test = x_test[:, :, off:, off:, off:]
y_train = y_train[:, :, off:, off:, off:]
y_test = y_test[:, :, off:, off:, off:]
y_train[y_train != 2] = 0
y_test[y_test != 2] = 0
y_train[y_train == 2] = 1
y_test[y_test == 2] = 1

input_dim = x_train.shape[1:]
shp = list(x_train.shape)
shp = shp[2:]

# encoding layer
conv_channel_1 = 8
conv_channel_2 = 20
kern_size = 3

############################ encoder - semantic decoder ##########################
input_patches = Input(shape=input_dim)
x0 = Conv3D(conv_channel_1, kernel_size=kern_size, input_shape=input_dim,
            data_format='channels_first', padding='same', activation='relu')(input_patches)
x1 = MaxPooling3D((2, 2, 2), data_format='channels_first')(x0)
x2 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(x1)
encoded = MaxPooling3D((2, 2, 2), data_format='channels_first')(x2)

x3 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(encoded)
x4 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x3)
x5 = Concatenate(axis=1)([x4, x2])
x6 = Conv3D(conv_channel_1, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(x5)
x7 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x6)
x8 = Concatenate(axis=1)([x7, x0])
decoded = Conv3D(1, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(x8)
encoder = Model(inputs=input_patches, outputs=encoded)
####################################################################################


# compile and fit model
decoder = Model(input_patches, decoded)
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.fit(x_train, y_train,
            epochs=40,
            batch_size=128,
            shuffle=True,
            verbose=2,
            validation_split=.25,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

encode_name = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_unet_encoder.h5'
encoder.save(encode_name)

rand_int1 = np.random.randint(0, x_test.shape[0])
ex_1 = x_test[rand_int1, :]
ex_1_label = y_test[rand_int1, :]
ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
ex_1_pred = decoder.predict(ex_1)
visualize_results(ex_1, ex_1_label, ex_1_pred, shp)


# if encoded available, check it out
# if encoded_and_decoded:
#     ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
#     ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
#     encoded_imgs = encoder.predict(ex_1)
#     # encoder.save(encode_name)

