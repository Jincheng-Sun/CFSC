from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keras import models
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix,f1_score,\
    average_precision_score,precision_score,recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score


class AssesmentROC:
    def __init__(self, model, X_test, Y_test):
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test

    def _load_dataset(self,change_shape = True):
        if change_shape:
            self.X_test = np.reshape(self.X_test, [4000, 100, 100, 1])
        self.y_test = label_binarize(self.Y_test, classes=[0, 1, 2, 3, 4])
        self.n_classes = self.y_test.shape[1]

    def _predict(self, predict_classes = False):

        self.y_score_class = self.model.predict_classes(self.X_test)
        self.y_score = self.model.predict(self.X_test)




    def _drawROC(self):
        lw = 1
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), self.y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Assesment module')
        plt.legend(loc="lower right")
        plt.show()

    def confusion_metrix(self):
        print(confusion_matrix(y_true=self.Y_test, y_pred=self.y_score_class,
                               labels=[0,1,2,3,4]))
    def score(self):

        test = self.Y_test.reshape([-1,1])
        score = self.y_score_class.reshape([-1,1])
        macro = f1_score(y_true=test, y_pred=score,average='macro')
        micro = f1_score(y_true=test, y_pred=score,average='micro')
        precision_a = precision_score(
            y_true=test, y_pred=score, average='macro')
        precision_i = precision_score(
            y_true=test, y_pred=score, average='micro')
        recall_a = recall_score(
            y_true=test, y_pred=score, average='macro')
        recall_i = recall_score(
            y_true=test, y_pred=score, average='micro')

        print(self.Y_test.shape)
        print(macro)
        print(precision_a)
        print(recall_a)
        print(micro)
        print(precision_i)
        print(recall_i)




model = models.load_model('../Training/80000NN')
X_test = np.load('../data/test_x.npy')
Y_test = np.load('../data/test_y.npy')
keras_ROC = AssesmentROC(model, X_test, Y_test)
keras_ROC._load_dataset(change_shape=False)
keras_ROC._predict()
keras_ROC._drawROC()
keras_ROC.confusion_metrix()
keras_ROC.score()
# # X_train reshape to [40000,100,100]
# X_test = np.reshape(X_test, [4000, 100, 100, 1])
# # Binarize the output
# y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
# n_classes = y_test.shape[1]
# y_score = model.predict(X_test)
# # Plot linewidth.
# lw = 1
#
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # Compute macro-average ROC curve and ROC area
#
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
# # Finally average it and compute AUC
# mean_tpr /= n_classes
#
# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
# # Plot all ROC curves
# plt.figure(1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=2)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=2)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Assesment module')
# plt.legend(loc="lower right")
# plt.show()
#

# # Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()
