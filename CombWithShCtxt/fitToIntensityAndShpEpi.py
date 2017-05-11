"""
    Here the saved model shall be loaded and fit to matlab data - EPI
"""

# import numpy as np
import os
import sys
import h5py

import numpy as np
from keras.models import load_model
from scipy.io import loadmat, savemat

import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

# the best model is always saved in the nn_matching_models folder
MODEL_NAME = '/home/nripesh/PycharmProjects/Siamese/CombWithShCtxt/int_shctxt_match_model_epi.h5'

src = sys.argv[1]
# src = '/home/nripesh/Dropbox/research_matlab/nripesh_3d/nbor_shp_data/'

rand_idf = sys.argv[2]

if os.path.isfile(src + 'patch_pairs_' + rand_idf + '_1.h5'):
    # only load if .mat file is found
    intensity_model = load_model(MODEL_NAME)

    no_of_files = int(sys.argv[3])
    print('Processing epi, total  of: ' + str(no_of_files) + ' files.')

    for i in range(no_of_files):
        # get patch data
        patch_data_file = src + "patch_pairs_" + str(rand_idf) + "_" + str(i + 1) + ".h5"
        size_file = src + "DIM_" + str(rand_idf) + "_" + str(i+1) + ".h5"

        with h5py.File(size_file, 'r') as hf:
            DIM = hf.get('DIM')
            DIM = np.array(DIM).astype('int')
            DIM = DIM[:, 0]
            # DIM = np.concatenate([DIM[0:2], [1], DIM[2:]])

        with h5py.File(patch_data_file, 'r') as hf:
            data = hf.get('patch_pairs')
            np_data = np.array(data).astype('float32')
        x_int_data = np.reshape(np_data, DIM)

        if x_int_data.max() > 1:
            x_int_data /= x_int_data.max()

        # get shctxt data
        patch_shp_file = src + "shctx_pairs_" + str(rand_idf) + "_" + str(i + 1) + ".h5"
        size_file = src + "SH_DIM_" + str(rand_idf) + "_" + str(i + 1) + ".h5"

        with h5py.File(size_file, 'r') as hf:
            DIM = hf.get('SH_DIM')
            DIM = np.array(DIM).astype('int')
            DIM = DIM[:, 0]
            # DIM = np.concatenate([DIM[0:2], [1], DIM[2:]])

        with h5py.File(patch_shp_file, 'r') as hf:
            data = hf.get('shctx_pairs')
            np_data = np.array(data).astype('float32')
        x_shp_data = np.reshape(np_data, DIM)
        x_shp_data[np.isnan(x_shp_data)] = 0

        if x_shp_data.max() > 1:
            x_shp_data /= x_shp_data.max()

        model_pred = intensity_model.predict([x_int_data[:, 0], x_int_data[:, 1], x_shp_data[:, 0], x_shp_data[:, 1]])

        x_out = {"pair_cost": model_pred}
        savemat(src + 'nbors_cost_' + rand_idf + '_' + str(i+1) + '.mat', x_out)
        print('match cost generated for frame: ' + str(i+1))
else:
    print('file not found')
    print('nbor_int_all_' + rand_idf + '_1.mat not found')


# use this for model filter visualizations
def visualize_filters(model):
    filter_wts = model.layers[2].layers[4].get_weights()

    # filter_wts[0] has conv filters, filter_wts[1] has the dot product vector
    conv_n = len(filter_wts[0])
    kern_size = filter_wts[0][0].shape

    conv_n_choose = np.random.randint(0, conv_n)
    print("filter chosen: " + str(conv_n_choose))
    flip_1 = np.random.randint(0, 2)
    conv_filt_0 = filter_wts[0][conv_n_choose, flip_1, :, :, 0]
    conv_filt_1 = filter_wts[0][conv_n_choose, flip_1, :, :, 1]
    conv_filt_2 = filter_wts[0][conv_n_choose, flip_1, :, :, 2]

    plt.figure(1)
    plt.imshow(conv_filt_0, interpolation='none', cmap='gray')
    plt.figure(2)
    plt.imshow(conv_filt_1, interpolation='none', cmap='gray')
    plt.figure(3)
    plt.imshow(conv_filt_2, interpolation='none', cmap='gray')
    plt.show()


# use this to visualize input data - if it matches MATLAB format or not
# use x_data before normalization
def visualize_data_input(x_data):
    # n_i = np.random.randint(0, x_data.shape[0])
    # n_z = np.random.randint(0, 11)

    n_i = 99
    n_z = 5
    a = x_data[n_i, 1, 0, :, :, n_z]
    b = x_data[n_i, 1, 1, :, :, n_z]

    plt.figure(1)
    plt.imshow(a, interpolation='none', cmap='gray')

    plt.figure(2)
    plt.imshow(b, interpolation='none', cmap='gray')
    plt.show()
