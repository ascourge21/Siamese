def siamese_euclidean(y_true, y_pred):
    a = y_pred[0::2]
    b = y_pred[1::2]
    diff = ((a - b) ** 2).sum(axis=1, keepdims=True)
    y_true = y_true[0::2]
    return ((diff - y_true)**2).mean()