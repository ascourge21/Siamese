import numpy as np
import h5py
from scipy.io import loadmat

src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/shortest_paths/shortest_paths_3d/nbor_shp_data/'
my_file = src + "patch_pairs_368693_1.h5"
size_file = src + "DIM368693_1.h5"
data_mat = loadmat(src + 'patch_pairs_368693_1.mat').get('patch_pairs').astype('float32')

with h5py.File(my_file, 'r') as hf:
    data = hf.get('patch_pairs')
    np_data = np.array(data).astype('float32')

with h5py.File(size_file, 'r') as hf:
    DIM = hf.get('DIM')
    DIM = np.array(DIM).astype('int')


np_data_rshp = np.reshape(np_data, DIM[:, 0])
print(np_data_rshp[1231, 0, 1, 4, 6])