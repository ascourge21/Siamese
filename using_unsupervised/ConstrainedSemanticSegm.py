"""
    Here we'll learn an intensity encoder, that learns to mimic the code from a segmentation encoder-decoder
"""

import numpy as np
# from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Convolution3D, \
    MaxPooling3D, Flatten, UpSampling3D
# from keras.regularizers import WeightRegularizer, l2
from keras.models import Model
from keras.callbacks import EarlyStopping
# from keras import backend as K


import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
# from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, log_loss
# from sklearn.cross_validation import train_test_split

from siamese_supervised import createShapeData
from face_siamese.SiameseFunctions import eucl_dist_output_shape, euclidean_distance

SMALL_CONST = 10 ** -10


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
    draw_patch_in_2d(np.reshape(input_label, (shp[0], shp[1], shp[2])))  # labels
    draw_patch_in_2d(np.reshape(pred_im, (shp[0], shp[1], shp[2])))  # predicted symantic labels
    plt.show()

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'leuven_labeled_semantic_patches_'

tr_id = [25, 26, 28, 29]
# tr_id = [26]
test_id = 27

# get paired dataset and remove the pairing
x_train, x_test, y_train, y_test = create_loo_train_test_set(src, data_stem, tr_id, test_id)

# randinds = np.random.randint(0, x_train.shape[0], 1000)
randinds = np.array(range(x_train.shape[0]))
off = 1
x_train = x_train[randinds, :, off:, off:, off:]
# x_train = x_train[:, :, off:, off:, off:]
x_test = x_test[:, :, off:, off:, off:]
y_train = y_train[randinds, :, off:, off:, off:]
y_test = y_test[:, :, off:, off:, off:]

# keep this simple - just do 0/1
y_train[y_train != 2] = 0
y_test[y_test != 2] = 0
y_train[y_train == 2] = 1
y_test[y_test == 2] = 1

# test visualize
shp = list(x_train.shape)
shp = shp[2:]
input_dim = x_train.shape[1:]
# visualize_results(x_train[0, 0, :, :, :], y_train[0, 0, :, :, :], y_train[0, 0, :, :, :], shp)


######################### encoding layer - segmentation #########################
conv_channel_1 = 5
conv_channel_2 = 15
kern_size = 3

segmen_patches = Input(shape=input_dim)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                  activation='relu', dim_ordering='th', border_mode='same')(segmen_patches)
x = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)
x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                  activation='relu', dim_ordering='th', border_mode='same')(x)
encoded_semantic = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)

x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(encoded_semantic)
x = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                  border_mode='same')(x)
x = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x)
segmen_recons = Convolution3D(1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                              border_mode='same')(x)

######################### encoding layer - intensity #########################
intensity_patches = Input(shape=input_dim)
x = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                  activation='relu', dim_ordering='th', border_mode='same')(intensity_patches)
x = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)
x = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                  activation='relu', dim_ordering='th', border_mode='same')(x)
encoded_intensity = MaxPooling3D((2, 2, 2), dim_ordering='th')(x)


############################# train and fit #####################################

segmentation_encoder = Model(input=segmen_patches, output=encoded_semantic)

# compile and fit model
segmen_encoder_decoder_model = Model(input=[segmen_patches],
                                     output=[segmen_recons])
segmen_encoder_decoder_model.compile(optimizer='RMSprop', loss='binary_crossentropy')
intensity_encoder_model = Model(input=[intensity_patches],
                                output=[encoded_intensity])
intensity_encoder_model.compile(optimizer='RMSprop', loss='mean_squared_error')


# first train each models for 1 epoch, separately
segmen_encoder_decoder_model.fit([y_train], [y_train], nb_epoch=1, batch_size=128, shuffle=True,
                                 verbose=2, validation_split=.25,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
encoded_seg_eval = segmentation_encoder.predict(y_train)
intensity_encoder_model.fit([x_train], encoded_seg_eval, nb_epoch=1, batch_size=128, shuffle=True,
                            verbose=2, validation_split=.25,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

# now construct a joint model and train it
batch_size = 128
dist_match_template = np.zeros((x_train.shape[0]))
encoded_intensity_flat = Flatten()(encoded_intensity)
encoded_semantic_flat = Flatten()(encoded_semantic)
encoding_distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_semantic_flat, encoded_intensity_flat])
intensity_segmen_joint_model = Model(input=[intensity_patches, segmen_patches],
                                     output=[segmen_recons, encoding_distance])
intensity_segmen_joint_model.compile(optimizer='RMSprop', loss=['binary_crossentropy', 'mean_squared_error'],
                                     loss_weights=[10, 1])
intensity_segmen_joint_model.fit([x_train, y_train], [y_train, dist_match_template], nb_epoch=40, batch_size=batch_size,
                                 shuffle=True, verbose=2, validation_split=.25,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])


# save the intensity encoder
encode_name = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_acnn_encoder.h5'
intensity_encoder_model.save(encode_name)

# segmentation reconstruction
for i in range(5):
    rand_int1 = np.random.randint(0, x_test.shape[0])
    int_patch1 = x_test[rand_int1, :]
    segmen_patch1 = y_test[rand_int1, :]
    ex_1_label = y_test[rand_int1, :]
    int_patch1 = np.reshape(int_patch1, (1, int_patch1.shape[0], int_patch1.shape[1], int_patch1.shape[2], int_patch1.shape[3]))
    segmen_patch1 = np.reshape(segmen_patch1, (1, segmen_patch1.shape[0], segmen_patch1.shape[1], segmen_patch1.shape[2], segmen_patch1.shape[3]))
    segmen_patch_recon = segmen_encoder_decoder_model.predict([segmen_patch1])
    # visualize_results(int_patch1, ex_1_label, segmen_patch_recon, shp)

    # now let's see if the two encoders produce similar results -> definitely didn't work
    int_code = intensity_encoder_model.predict(int_patch1)
    seg_code = segmentation_encoder.predict(segmen_patch1)

    plt.figure(i)
    plt.plot(int_code.ravel())
    plt.hold(True)
    plt.plot(seg_code.ravel())
    plt.hold(False)

plt.show()

