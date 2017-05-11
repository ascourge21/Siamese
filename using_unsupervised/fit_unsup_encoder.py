"""
    here we'll call the encoder and fit on the encoder.
"""

# import numpy as np
import os
import sys
import h5py

import matplotlib
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

import numpy as np
from keras.models import load_model
from scipy.io import loadmat, savemat

# the best model is always saved in the nn_matching_models folder
MODEL_NAME = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_int_encoder.h5'
# MODEL_NAME = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_semi_sup_encoder.h5'

# extract command line inputs
src = sys.argv[1]
rand_idf = sys.argv[2]
no_of_files = int(sys.argv[3])

# src = '/home/nripesh/Dropbox/research_matlab/nripesh_3d/nbor_shp_data/'


# here we'll try to do robust distance calculation
def dist_calc_simple(x_0, x_1):
    model_pred = np.zeros((x_0.shape[0], 1))
    for k in range(model_pred.shape[0]):
        model_pred[k] = np.linalg.norm(x_0[k, :] - x_1[k, :])/x_0.shape[1]
    return model_pred


# here we'll try to do robust distance calculation
def dist_calc(x_0, x_1):
    diff_data = x_0 - x_1
    cov_mat = np.cov(diff_data.T)
    cov_mat_inv_pseudo = np.linalg.inv(cov_mat)  # first let's just try normal one without being fancy
    # mahab_dist = np.zeros((diff_data.shape[0], 1))

    # cov_mat_inv_pseudo = np.linalg.pinv(cov_mat, rcond=1.5e-2)
    mahab_dist_pseudo = np.zeros((diff_data.shape[0], 1))

    for j in range(diff_data.shape[0]):
        dx = np.reshape(diff_data[j, :], (diff_data.shape[1], 1))
        # mahab_dist[i] = np.sqrt(np.dot(dx.T, np.dot(cov_mat_inv, dx)) / diff_data.shape[1])
        mahab_dist_pseudo[j] = np.sqrt(np.dot(dx.T, np.dot(cov_mat_inv_pseudo, dx)) / diff_data.shape[1])

    draw_figs = 0
    if draw_figs:
        # print('diff between orig and smoothed: ' + str(np.sum(np.abs(mahab_dist - mahab_dist_pseudo))))

        plt.figure(2)
        plt.plot(model_pred, 'g.')
        # plt.figure(3)
        # plt.plot(mahab_dist, 'r.')
        plt.figure(4)
        plt.plot(mahab_dist_pseudo, 'b.')
        plt.show()

    return mahab_dist_pseudo


# run the model
if os.path.isfile(src + 'patch_pairs_' + rand_idf + '_1.h5'):
    # only load if .mat file is found
    intensity_encoder = load_model(MODEL_NAME)

    print('Processing endo, total  of: ' + str(no_of_files) + ' files.')

    for i in range(no_of_files):
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
        x_data = np.reshape(np_data, DIM)

        if x_data.max() > 1:
            x_data /= 255

        x_0_encode = intensity_encoder.predict(x_data[:, 0, :, 1:, 1:, 1:])
        x_1_encode = intensity_encoder.predict(x_data[:, 1, :, 1:, 1:, 1:])

        # vectorize the matrices
        x_en_sz = x_0_encode.shape
        x_0_encode = np.reshape(x_0_encode, (x_en_sz[0], x_en_sz[1]*x_en_sz[2]*x_en_sz[3]*x_en_sz[4]))
        x_1_encode = np.reshape(x_1_encode, (x_en_sz[0], x_en_sz[1]*x_en_sz[2]*x_en_sz[3]*x_en_sz[4]))

        # model_pred = np.zeros((x_0_encode.shape[0], 1))
        # for k in range(model_pred.shape[0]):
        #     model_pred[k] = np.linalg.norm(x_0_encode[k, :] - x_1_encode[k, :])/x_0_encode.shape[1]

        model_pred = dist_calc_simple(x_0_encode, x_1_encode)

        draw_figs = 0
        if draw_figs:
            plt.figure(1)
            plt.plot(x_0_encode.max(axis=1), 'r.')
            plt.plot(x_1_encode.max(axis=1), 'b.')
            plt.plot(model_pred, 'g.')
            plt.show()

        x_out = {"pair_cost": model_pred}
        savemat(src + 'nbors_cost_' + rand_idf + '_' + str(i+1) + '.mat', x_out)
        print('match cost generated for frame: ' + str(i+1))
else:
    print('file not found')
    print('nbor_int_all_' + rand_idf + '_1.mat not found')



