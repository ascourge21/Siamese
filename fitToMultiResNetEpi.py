"""
    Here the saved model shall be loaded and fit to matlab data - EPI
"""

import os
import sys
import h5py
import numpy as np

from keras.models import load_model
from scipy.io import savemat


def get_dim(size_file):
    with h5py.File(size_file, 'r') as hf:
        dim = hf.get('DIM')
        dim = np.array(dim).astype('int')
        dim = dim[:, 0]
        dim = np.concatenate([dim[0:2], [1], dim[2:]])
    return dim


def get_data(patch_data_file, size_file):
    dim = get_dim(size_file)
    # print(dim)
    with h5py.File(patch_data_file, 'r') as hf:
        data = hf.get('patch_pairs')
        np_data = np.array(data).astype('float32')
    x_data = np.reshape(np_data, dim)
    return x_data


def get_dim_flat(size_file):
    with h5py.File(size_file, 'r') as hf:
        dim = hf.get('DIM')
        dim = np.array(dim).astype('int')
        dim = dim[:, 0]
        sz = [dim[0], dim[1], dim[2]*dim[3]*dim[4]]
    return sz


def get_data_flat(patch_data_file, size_file):
    dim = get_dim_flat(size_file)
    # print(dim)
    with h5py.File(patch_data_file, 'r') as hf:
        data = hf.get('patch_pairs')
        np_data = np.array(data).astype('float32')
    x_data = np.reshape(np_data, dim)
    return x_data

MODEL_NAME = '/home/nripesh/Dropbox/research_matlab/nripesh_3d/nn_matching_models/shape_match_model_epi_multi_res.h5'

src = sys.argv[1]
# src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/shortest_paths/shortest_paths_3d/nbor_shp_data/'

rand_idf = sys.argv[2]

if os.path.isfile(src + 'patch_pairs_lg_' + rand_idf + '_1.h5'):
    # only load if .mat file is found
    intensity_model = load_model(MODEL_NAME)

    no_of_files = int(sys.argv[3])
    print('Processing epi, total  of: ' + str(no_of_files) + ' files.')

    for i in range(no_of_files):
        # large 4D patch
        large_patch_file = src + "patch_pairs_lg_" + str(rand_idf) + "_" + str(i + 1) + ".h5"
        large_dim_file = src + "DIM_lg_" + str(rand_idf) + "_" + str(i+1) + ".h5"
        x_data_lg = get_data(large_patch_file, large_dim_file)

        if x_data_lg.max() > 1:
            x_data_lg /= 255

        # small flat patch
        small_patch_file = src + "patch_pairs_sm_" + str(rand_idf) + "_" + str(i + 1) + ".h5"
        small_dim_file = src + "DIM_sm_" + str(rand_idf) + "_" + str(i+1) + ".h5"
        x_data_sm = get_data_flat(small_patch_file, small_dim_file)

        if x_data_sm.max() > 1:
            x_data_sm /= 255

        # predict
        model_pred = intensity_model.predict([x_data_lg[:, 0], x_data_lg[:, 1], x_data_sm[:, 0], x_data_sm[:, 1]])

        x_out = {"pair_cost": model_pred}
        savemat(src + 'nbors_cost_' + rand_idf + '_' + str(i+1) + '.mat', x_out)
        print('match cost generated for frame: ' + str(i+1))
else:
    print('file not found')
