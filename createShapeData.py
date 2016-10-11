"""
    here i'll load my matlab shape ctxt data (r, theta, phi) and try to run it through a siamese network
"""

import numpy as np
import pandas as pd

from scipy.io import loadmat, savemat


def get_shape_data(train_pct):
    # total length of data should be 2* (24,000 + 46,000) = 140,000
    src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/matconvnet-1.0-beta21/cardiac_data/'
    x_match_a = pd.read_csv(src + 'X_match_a.csv', header=None).as_matrix().astype('float32')
    x_match_b = pd.read_csv(src + 'X_match_b.csv', header=None).as_matrix().astype('float32')
    x_non_match_a = pd.read_csv(src + 'X_non_match_a.csv', header=None).as_matrix().astype('float32')
    x_non_match_b = pd.read_csv(src + 'X_non_match_b.csv', header=None).as_matrix().astype('float32')

    # get indices and lengths for the split
    x_m_inds = np.arange(x_match_a.shape[0])
    np.random.shuffle(x_m_inds)
    x_m_train_len = int(train_pct*x_match_a.shape[0])
    x_m_test_len = x_match_a.shape[0] - x_m_train_len

    x_n_inds = np.arange(x_non_match_a.shape[0])
    np.random.shuffle(x_n_inds)
    x_n_train_len = int(train_pct*x_non_match_a.shape[0])
    x_n_test_len = x_non_match_a.shape[0] - x_n_train_len

    # get the train set first
    x_train = np.zeros([x_m_train_len*2 + x_n_train_len*2, x_match_a.shape[1]])
    y_train = np.zeros([x_m_train_len*2 + x_n_train_len*2, 1])
    count1 = 0
    for i in range(x_m_train_len):
        x_train[count1, :] = x_match_a[x_m_inds[i], :]
        x_train[count1+1, :] = x_match_b[x_m_inds[i], :]
        y_train[count1] = 1
        y_train[count1 + 1] = 1
        count1 += 2

    for i in range(x_n_train_len):
        x_train[count1, :] = x_non_match_a[x_n_inds[i], :]
        x_train[count1+1, :] = x_non_match_b[x_n_inds[i], :]
        y_train[count1] = 0
        y_train[count1 + 1] = 0
        count1 += 2

    # the test set
    x_test = np.zeros([x_m_test_len*2 + x_n_test_len*2, x_match_a.shape[1]])
    y_test = np.zeros([x_m_test_len*2 + x_n_test_len*2, 1])
    count2 = 0
    for i in range(x_m_test_len):
        x_test[count2, :] = x_match_a[x_m_inds[x_m_train_len + i], :]
        x_test[count2 + 1, :] = x_match_b[x_m_inds[x_m_train_len + i], :]
        y_test[count2] = 1
        y_test[count2 + 1] = 1
        count2 += 2

    for i in range(x_n_test_len):
        x_test[count2, :] = x_non_match_a[x_n_inds[x_n_train_len + i], :]
        x_test[count2 + 1, :] = x_non_match_b[x_n_inds[x_n_train_len + i], :]
        y_test[count2] = 0
        y_test[count2 + 1] = 0
        count2 += 2

    return x_train, x_test, y_train, y_test


def get_shape_data_paired_format():
    # total length of data should be 2* (24,000 + 46,000) = 140,000
    src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/matconvnet-1.0-beta21/cardiac_data/'
    x_match_a = pd.read_csv(src + 'X_match_a.csv', header=None).as_matrix().astype('float32')
    x_match_b = pd.read_csv(src + 'X_match_b.csv', header=None).as_matrix().astype('float32')
    x_non_match_a = pd.read_csv(src + 'X_non_match_a.csv', header=None).as_matrix().astype('float32')
    x_non_match_b = pd.read_csv(src + 'X_non_match_b.csv', header=None).as_matrix().astype('float32')

    x_match = np.concatenate([x_match_a, x_match_b], axis=1)
    x_match = np.reshape(x_match, [x_match.shape[0], 2, int(x_match.shape[1]/2)])
    y_match = np.ones([x_match.shape[0], 1])

    x_non_match = np.concatenate([x_non_match_a, x_non_match_b], axis=1)
    x_non_match = np.reshape(x_non_match, [x_non_match.shape[0], 2, int(x_non_match.shape[1] / 2)])
    y_non_match = np.zeros([x_non_match.shape[0], 1])

    x_out = np.concatenate([x_match, x_non_match]).astype('float32')
    y_out = np.concatenate([y_match, y_non_match]).astype('float32')

    return x_out, y_out


def get_int_paired_format(src, data_name):
    # total length of data should be 2* (24,000 + 46,000) = 140,000

    shape_data = loadmat(src + data_name)
    x_match_a = shape_data.get('X_match_a').astype('float32')
    x_match_b = shape_data.get('X_match_b').astype('float32')
    x_non_match_a = shape_data.get('X_non_match_a').astype('float32')
    x_non_match_b = shape_data.get('X_non_match_b').astype('float32')

    x_match_a = x_match_a.reshape([x_match_a.shape[0], 1, 1, x_match_a.shape[1], x_match_a.shape[2], x_match_a.shape[3]])
    x_match_b = x_match_b.reshape([x_match_b.shape[0], 1, 1, x_match_b.shape[1], x_match_b.shape[2], x_match_b.shape[3]])
    x_match = np.concatenate([x_match_a, x_match_b], axis=1)
    y_match = np.ones([x_match.shape[0], 1])

    x_non_match_a = x_non_match_a.reshape([x_non_match_a.shape[0], 1, 1, x_non_match_a.shape[1], x_non_match_a.shape[2],
                                           x_non_match_a.shape[3]])
    x_non_match_b = x_non_match_b.reshape([x_non_match_b.shape[0], 1, 1, x_non_match_b.shape[1], x_non_match_b.shape[2],
                                           x_non_match_b.shape[3]])
    x_non_match = np.concatenate([x_non_match_a, x_non_match_b], axis=1)
    y_non_match = np.zeros([x_non_match.shape[0], 1])

    x_out = np.concatenate([x_match, x_non_match]).astype('float32')
    y_out = np.concatenate([y_match, y_non_match]).astype('float32')

    return x_out, y_out


def get_int_paired_format_flattened(src, data_name):
    # total length of data should be 2* (24,000 + 46,000) = 140,000

    shape_data = loadmat(src + data_name)
    x_match_a = shape_data.get('X_match_a').astype('float32')
    x_match_b = shape_data.get('X_match_b').astype('float32')
    x_non_match_a = shape_data.get('X_non_match_a').astype('float32')
    x_non_match_b = shape_data.get('X_non_match_b').astype('float32')

    x_match_a = x_match_a.reshape([x_match_a.shape[0], 1, x_match_a.shape[1]*x_match_a.shape[2]*x_match_a.shape[3]])
    x_match_b = x_match_b.reshape([x_match_b.shape[0], 1, x_match_b.shape[1]*x_match_b.shape[2]*x_match_b.shape[3]])
    x_match = np.concatenate([x_match_a, x_match_b], axis=1)
    y_match = np.ones([x_match.shape[0], 1])

    x_non_match_a = x_non_match_a.reshape([x_non_match_a.shape[0], 1,
                                           x_non_match_a.shape[1]*x_non_match_a.shape[2]*x_non_match_a.shape[3]])
    x_non_match_b = x_non_match_b.reshape([x_non_match_b.shape[0], 1,
                                           x_non_match_b.shape[1]*x_non_match_b.shape[2]*x_non_match_b.shape[3]])
    x_non_match = np.concatenate([x_non_match_a, x_non_match_b], axis=1)
    y_non_match = np.zeros([x_non_match.shape[0], 1])

    x_out = np.concatenate([x_match, x_non_match]).astype('float32')
    y_out = np.concatenate([y_match, y_non_match]).astype('float32')

    return x_out, y_out
