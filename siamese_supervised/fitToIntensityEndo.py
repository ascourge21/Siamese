"""plt
    Here the saved model shall be loaded and fit to matlab data - ENDO
"""

# import numpy as np
import os
import sys
import h5py

import numpy as np
from keras.models import load_model
from scipy.io import loadmat, savemat
from SiameseFunctions import contrastive_loss

# the best model is always saved in the nn_matching_models folder
# MODEL_NAME = 'shape_match_model_endo_k3.h5'
MODEL_NAME = 'shape_match_model_endo_k3_new.h5'


src = sys.argv[1]
# src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/shortest_paths/shortest_paths_3d/nbor_shp_data/'

rand_idf = sys.argv[2]

if os.path.isfile(src + 'patch_pairs_' + rand_idf + '_1.h5'):
    # only load if .mat file is found
    intensity_model = load_model(MODEL_NAME, custom_objects={'contrastive_loss': contrastive_loss})

    no_of_files = int(sys.argv[3])
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

        model_pred = intensity_model.predict([x_data[:, 0], x_data[:, 1]])

        x_out = {"pair_cost": model_pred}
        savemat(src + 'nbors_cost_' + rand_idf + '_' + str(i+1) + '.mat', x_out)
        print('match cost generated for frame: ' + str(i+1))
else:
    print('file not found')
    print('nbor_int_all_' + rand_idf + '_1.mat not found')


