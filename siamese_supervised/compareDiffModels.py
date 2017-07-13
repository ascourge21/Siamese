"""
    here we'll compare the supervised, unsupervised and the transfer learning methods - using AUC
"""

import matplotlib
import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_curve, auc

from siamese_supervised import createShapeData

matplotlib.use('qt4agg')
from matplotlib import pyplot as plt


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

    return mahab_dist_pseudo


# create groups of 4 image sets as training and 1 as test
def create_loo_train_test_set(src, data_stem, train_ids, test_id):
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


def test_on_SUP_model(sup_model_name, x, y):
    SUP_MODEL = load_model(sup_model_name)
    model_pred = SUP_MODEL.predict([x[:, 0], x[:, 1]])
    tpr, fpr, _ = roc_curve(y, model_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.hold(True)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.hold(False)
    plt.savefig('/home/nripesh/Dropbox/temp_images/nnet_train_images/roc_curve_siamese.png')


def test_on_UNSUP_model(unsup_model_name, x, y):
    UNSUP_ENCODER = load_model(unsup_model_name)
    x_0_encode = UNSUP_ENCODER.predict(x[:, 0, :, 1:, 1:, 1:])
    x_1_encode = UNSUP_ENCODER.predict(x[:, 1, :, 1:, 1:, 1:])
    # vectorize the matrices
    x_en_sz = x_0_encode.shape
    x_0_encode = np.reshape(x_0_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))
    x_1_encode = np.reshape(x_1_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))
    model_pred = dist_calc(x_0_encode, x_1_encode)
    tpr, fpr, _ = roc_curve(y, model_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(2)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.hold(True)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.hold(False)
    plt.savefig('/home/nripesh/Dropbox/temp_images/nnet_train_images/roc_curve_autoencoder.png')


def test_on_SEMISUP_SEMANTIC_model(semantic_segmentation_model_name, x, y):
    SEMANTIC_ENCODER = load_model(semantic_segmentation_model_name)
    x_0_encode = SEMANTIC_ENCODER.predict(x[:, 0, :, 1:, 1:, 1:])
    x_1_encode = SEMANTIC_ENCODER.predict(x[:, 1, :, 1:, 1:, 1:])
    # vectorize the matrices
    x_en_sz = x_0_encode.shape
    x_0_encode = np.reshape(x_0_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))
    x_1_encode = np.reshape(x_1_encode, (x_en_sz[0], x_en_sz[1] * x_en_sz[2] * x_en_sz[3] * x_en_sz[4]))

    # plot a random code
    for i in range(5):
        rand_int = np.random.randint(0, x_0_encode.shape[0])
        print('y is: ' + str(y[rand_int]))
        plt.figure(i)
        plt.plot(x_0_encode[rand_int, :].ravel())
        plt.hold(True)
        plt.plot(x_1_encode[rand_int, :].ravel())
        plt.hold(False)
    plt.show()

    model_pred = dist_calc_simple(x_0_encode, x_1_encode)
    tpr, fpr, _ = roc_curve(y, model_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(3)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.hold(True)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.hold(False)
    plt.savefig('/home/nripesh/Dropbox/temp_images/nnet_train_images/roc_curve_semantic_segmentation.png')


src = '/home/nripesh/Dropbox/research_matlab/feature_tracking/generating_train_data_forNNet/'
data_stem = 'x_data_intensity_comb_'

# tr_id = [1, 2, 3, 4, 5]  # too large with all of them
tr_id = [2]
x_train, _, y_train, _ = create_loo_train_test_set(src, data_stem, tr_id, 1)

# # siamese - supervised
# SUP_MODEL_NAME = 'shape_match_model1.h5'
# test_on_SUP_model(SUP_MODEL_NAME, x_train, y_train)
#
# # autoencoder - unsupervised
# UNSUP_MODEL_NAME = 'leuven_int_encoder.h5'
# test_on_UNSUP_model(UNSUP_MODEL_NAME, x_train, y_train)

# semantic - semi-supervised/transfer learning
SEMANTIC_MODEL_NAME = '/home/nripesh/PycharmProjects/Siamese/using_unsupervised/leuven_acnn_encoder.h5'
test_on_SEMISUP_SEMANTIC_model(SEMANTIC_MODEL_NAME, x_train, y_train)

