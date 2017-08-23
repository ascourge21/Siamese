"""
    Here we'll learn an intensity encoder, that learns to mimic the code from a segmentation encoder-decoder
"""

import numpy as np
# from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Conv3D, \
    MaxPooling3D, Flatten, UpSampling3D, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
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


def create_loo_train_test_set_int_paired(src, data_stem, train_ids, test_id):
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
    for j in range(4):
        axarr[0, j].imshow(im[:, :, j * 3], interpolation='none', cmap='gray')
        axarr[0, j].set_title('xy ' + str(j * 3))

    # xz
    for j in range(4):
        axarr[1, j].imshow(im[:, j * 3, :], interpolation='none', cmap='gray')
        axarr[1, j].set_title('xz ' + str(j * 3))

    # yz
    for j in range(4):
        axarr[2, j].imshow(im[j * 3, :, :], interpolation='none', cmap='gray')
        axarr[2, j].set_title('yz ' + str(j * 3))


def visualize_results(input_im, input_label, pred_im, shp):
    draw_patch_in_2d(np.reshape(input_im, (shp[0], shp[1], shp[2])))  # intensity
    draw_patch_in_2d(np.reshape(input_label, (shp[0], shp[1], shp[2])))  # labels
    draw_patch_in_2d(np.reshape(pred_im, (shp[0], shp[1], shp[2])))  # predicted symantic labels
    plt.show()


def dist_calc_simple(x_0, x_1):
    model_pred = np.zeros((x_0.shape[0], 1))
    for k in range(model_pred.shape[0]):
        model_pred[k] = np.linalg.norm(x_0[k, :] - x_1[k, :]) / x_0.shape[1]
    return model_pred

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'leuven_labeled_semantic_patches_'

# tr_id = [25, 26, 28, 29]
tr_id = [251]
test_id = 252

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


######################### encoding/decoding - segmentation #########################
conv_channel_1 = 8
conv_channel_2 = 20
kern_size = 3

segmen_patches = Input(shape=input_dim)
x0 = Conv3D(conv_channel_1, kernel_size=kern_size, input_shape=input_dim,
            data_format='channels_first', padding='same', activation='relu')(segmen_patches)
x1 = MaxPooling3D((2, 2, 2), data_format='channels_first')(x0)
x2 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(x1)
encoded_semantic = MaxPooling3D((2, 2, 2), data_format='channels_first')(x2)

x3 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first',
            padding='same', activation='relu')(encoded_semantic)
x4 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x3)
x5 = Concatenate(axis=1)([x4, x2])
x6 = Conv3D(conv_channel_1, kernel_size=kern_size, data_format='channels_first', padding='same', activation='relu')(x5)
x7 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x6)
x8 = Concatenate(axis=1)([x7, x0])
segmen_recons = Conv3D(1, kernel_size=kern_size, data_format='channels_first', padding='same', activation='sigmoid')(x8)

######################### encoding/decoding - intensity #########################
intensity_patches = Input(shape=input_dim)
x00 = Conv3D(conv_channel_1, kernel_size=kern_size, input_shape=input_dim,
             data_format='channels_first', padding='same', activation='relu')(intensity_patches)
x01 = MaxPooling3D((2, 2, 2), data_format='channels_first')(x00)
x02 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first',
             padding='same', activation='relu')(x01)
encoded_intensity = MaxPooling3D((2, 2, 2), data_format='channels_first')(x02)

x03 = Conv3D(conv_channel_2, kernel_size=kern_size, data_format='channels_first',
             padding='same', activation='relu')(encoded_intensity)
x04 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x03)
x05 = Concatenate(axis=1)([x04, x02])
x06 = Conv3D(conv_channel_1, kernel_size=kern_size, data_format='channels_first',
             padding='same', activation='relu')(x05)
x07 = UpSampling3D(size=(2, 2, 2), data_format='channels_first')(x06)
x08 = Concatenate(axis=1)([x07, x00])
intensity_recons = Conv3D(1, kernel_size=kern_size, data_format='channels_first',
                          padding='same', activation='sigmoid')(x08)


############################# train and fit #####################################
intensity_encoder = Model(inputs=[intensity_patches], outputs=[encoded_intensity])
segmentation_encoder = Model(inputs=[segmen_patches], outputs=[encoded_semantic])

# compile and fit model
segmen_encoder_decoder_model = Model(inputs=[segmen_patches], outputs=[segmen_recons])
segmen_encoder_decoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')
intensity_encoder_decoder_model = Model(inputs=[intensity_patches], outputs=[encoded_intensity, intensity_recons])
intensity_encoder_decoder_model.compile(optimizer='adadelta', loss=['mean_absolute_error', 'binary_crossentropy'])


# first train each models for 1 epoch, separately
segmen_encoder_decoder_model.fit([y_train], [y_train], epochs=1, batch_size=128, shuffle=True,
                                 verbose=2, validation_split=.25,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])
encoded_seg_eval = segmentation_encoder.predict(y_train)
intensity_encoder_decoder_model.fit([x_train], [encoded_seg_eval, y_train], epochs=1, batch_size=128, shuffle=True,
                                    verbose=2, validation_split=.25,
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

# now construct a joint model and train it
batch_size = 128
dist_match_template = np.zeros((x_train.shape[0]))
encoded_intensity_flat = Flatten()(encoded_intensity)
encoded_semantic_flat = Flatten()(encoded_semantic)
encoding_distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)\
    ([encoded_semantic_flat, encoded_intensity_flat])


# somehow first training without the intensity reconstruction seems to work well
intensity_segmen_joint_model = Model(inputs=[intensity_patches, segmen_patches],
                                     outputs=[segmen_recons, encoding_distance])
intensity_segmen_joint_model.compile(optimizer='adadelta',
                                     loss=['binary_crossentropy', 'mean_absolute_error'],
                                     loss_weights=[1, 1])
intensity_segmen_joint_model.fit([x_train, y_train], [y_train, dist_match_template],
                                 epochs=2, batch_size=batch_size,
                                 shuffle=True, verbose=2, validation_split=.25,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

####
intensity_segmen_joint_model = Model(inputs=[intensity_patches, segmen_patches],
                                     outputs=[segmen_recons, intensity_recons, encoding_distance])
intensity_segmen_joint_model.compile(optimizer='adadelta',
                                     loss=['binary_crossentropy', 'binary_crossentropy', 'mean_absolute_error'],
                                     loss_weights=[1, 1, 1])
intensity_segmen_joint_model.fit([x_train, y_train], [y_train, y_train, dist_match_template],
                                 epochs=2, batch_size=batch_size,
                                 shuffle=True, verbose=2, validation_split=.25,
                                 callbacks=[EarlyStopping(monitor='val_loss', patience=2)])


# save the intensity encoder
encode_name = 'leuven_acnn_encoder.h5'
intensity_encoder.save(encode_name)


def test_on_UNSUP_model(UNSUP_ENCODER, x, y):
    x_0_encode = UNSUP_ENCODER.predict(x[:, 0, :, 1:, 1:, 1:])
    x_1_encode = UNSUP_ENCODER.predict(x[:, 1, :, 1:, 1:, 1:])
    # vectorize the matrices
    x_en_sz = x_0_encode.shape
    x_0_encode = np.reshape(x_0_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))
    x_1_encode = np.reshape(x_1_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))
    model_pred = dist_calc_simple(x_0_encode, x_1_encode)
    tpr, fpr, _ = roc_curve(y, model_pred)
    roc_auc = auc(fpr, tpr)
    print('auc is : ' + str(roc_auc))

test_data_stem = 'x_data_intensity_comb_'
x_train, _, y_train, _ = create_loo_train_test_set_int_paired(src, test_data_stem, [2], 2)
test_on_UNSUP_model(intensity_encoder, x_train, y_train)


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
    int_code = intensity_encoder.predict(int_patch1)
    seg_code = segmentation_encoder.predict(segmen_patch1)

    plt.figure(i)
    plt.plot(int_code.ravel())
    plt.hold(True)
    plt.plot(seg_code.ravel())
    plt.hold(False)

plt.show()

