import numpy as np


def siamese_euclidean(y_true, y_pred):
    a = y_pred[0::2]
    b = y_pred[1::2]
    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
    y_true = y_true[0::2]
    return ((diff - y_true)**2).mean()

def contrastive_loss(y_true, y_pred):
    a = y_pred[0::2]
    b = y_pred[1::2]
    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
    y_true = y_true[0::2]
    margin = 1
    comp_mat = np.zeros([diff.shape[0], 2])
    comp_mat[:, 0] = (margin - diff).ravel()
    comp_mat[:, 1] = np.zeros([diff.shape[0], ])
    return (y_true*diff + (1-y_true)*np.amax(comp_mat, 1)).mean()
