"""
    Here the saved model shall be loaded and fit to matlab data
"""

# import numpy as np
import pickle
import os
import sys

from scipy.io import loadmat, savemat

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/shortest_paths/shortest_paths_3d/nbor_shp_data/'

if os.path.isfile(src + '/nbor_shp_all_1.mat'):
    # only load if .mat file is found
    shape_model = pickle.load(open('shape_match_model.pl', 'rb'))
    no_of_files = int(sys.argv[1])

    for i in range(no_of_files):
        shape_data = loadmat(src+'/nbor_shp_all_' + str(i+1) + '.mat')
        x_data = shape_data.get('nbor_shp_all').astype('float32')

        model_pred = shape_model.predict([x_data[:, 0], x_data[:, 1]])

        x_out = {"pair_cost": model_pred}
        savemat(src+'/nbors_cost_' + str(i+1) + '.mat', x_out)
        print('shape cost generated for frame: ' + str(i+1))
else:
    print('nbor_shp_all_1.mat not found')

