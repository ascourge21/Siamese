"""
    this shall be used to load the labeled data
"""


from scipy.io import loadmat
from keras.utils import np_utils


def get_labeled_patches(src, data_name):
    # total length of data should be 2* (24,000 + 46,000) = 140,000
    # src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/' + \
    #       'generating_train_data_forNNet/dsea_data_based_train_patches/'
    # data_name = 'dsea_labeled_patches_74'

    print('loading... ' + data_name)
    shape_data = loadmat(src + data_name)
    x_patch = shape_data.get('X_patch').astype('float32')
    y = shape_data.get('labels').astype('int8')
    y = np_utils.to_categorical(y-1)

    if x_patch.max() > 1:
        x_patch /= 255

    return x_patch, y
