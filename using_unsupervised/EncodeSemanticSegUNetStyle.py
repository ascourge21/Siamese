"""
    Here, we'll learn the relationship between the
        input - image patches and the
        output - semantic segmentation labeling.
    We'll use the same code as the auto-encoding, but instead of reconstructing the input,
        we'll reconstruct the segmentation.
"""



import numpy as np
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Lambda
from keras.layers import Input, Dense, Dropout, Convolution3D, \
    MaxPooling3D, Flatten, BatchNormalization, UpSampling3D, Merge
from keras.regularizers import WeightRegularizer, l2
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras import initializations

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.cross_validation import train_test_split

import createShapeData
from SiameseFunctions import create_base_network, eucl_dist_output_shape, euclidean_distance, \
    contrastive_loss


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

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'leuven_labeled_semantic_patches_'

# tr_id = [25, 26, 28, 29]
tr_id = [25]
test_id = 27

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

############################ encoder - semantic decoder ##########################
input_patches = Input(shape=input_dim)
x0 = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, input_shape=input_dim,
                   activation='relu', dim_ordering='th', border_mode='same')(input_patches)
x1 = MaxPooling3D((2, 2, 2), dim_ordering='th')(x0)
x2 = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size,
                   activation='relu', dim_ordering='th', border_mode='same')(x1)
encoded = MaxPooling3D((2, 2, 2), dim_ordering='th')(x2)

x3 = Convolution3D(conv_channel_2, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                   border_mode='same')(encoded)
x4 = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x3)
x4 = Merge(mode='concat', concat_axis=1)([x4, x2])
x5 = Convolution3D(conv_channel_1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                   border_mode='same')(x4)
x6 = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(x5)
x4 = Merge(mode='concat', concat_axis=1)([x6, x0])
decoded = Convolution3D(1, kern_size, kern_size, kern_size, activation='relu', dim_ordering='th',
                        border_mode='same')(x6)
encoder = Model(input=input_patches, output=encoded)
####################################################################################


# compile and fit model
decoder = Model(input_patches, decoded)
decoder.compile(optimizer='adadelta', loss='mean_absolute_error')
decoder.fit(x_train, y_train,
            nb_epoch=40,
            batch_size=128,
            shuffle=True,
            verbose=2,
            validation_split=.25,
            callbacks=[EarlyStopping(monitor='val_loss', patience=2)])

encode_name = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_semi_sup_encoder.h5'
encoder.save(encode_name)

rand_int1 = np.random.randint(0, x_test.shape[0])
ex_1 = x_test[rand_int1, :]
ex_1_label = y_test[rand_int1, :]
ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
ex_1_pred = decoder.predict(ex_1)
visualize_results(ex_1, ex_1_label, ex_1_pred)


# if encoded available, check it out
# if encoded_and_decoded:
#     ex_1 = x_test[np.random.randint(0, x_test.shape[0]), :]
#     ex_1 = np.reshape(ex_1, (1, ex_1.shape[0], ex_1.shape[1], ex_1.shape[2], ex_1.shape[3]))
#     encoded_imgs = encoder.predict(ex_1)
#     # encoder.save(encode_name)

