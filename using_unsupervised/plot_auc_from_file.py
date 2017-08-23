import matplotlib
from sklearn.metrics import roc_curve, auc
matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

with open('/home/nripesh/PycharmProjects/Siamese/using_unsupervised/crossval_results_witch_auc.txt') as f:
    lines = f.readlines()
    tpr, fpr = [], []
    for line in lines:
        l = line.strip().split(', ')
        tpr.append(float(l[0]))
        fpr.append(float(l[1]))

    roc_auc = auc(fpr, tpr)
    print('auc is : ' + str(roc_auc))
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
    plt.savefig('cross_val_siamese.png')

