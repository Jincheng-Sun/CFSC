from model_adaptor import ModelAdaptor
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
import os


class AssessModel:
	"""
	model_adaptor: model specific adaptor
	classes: list of all classes
	y_pred_score: n * classes_num predict score

	"""
	def __init__(self, model_adaptor, classes):
		self.model_adaptor = model_adaptor
		self.classes = classes


	def draw_roc(self):
		y_pred_score = self.model_adaptor.get_pred_score()
		Y_test = self.model_adaptor.get_Y()

		Y_test_onehot = label_binarize(Y_test, classes=self.classes)
		n_classes = len(classes)

		fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # change threshold to collect fpr,tpr dots and AUC for each classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test_onehot[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # calculate fpr, tpr dots and AUC with average methord of micro
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_onehot.ravel(), y_pred_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        # calculate fpr, tpr dots and AUC with average methord of micro
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
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

        colors = cycle(['cyan', 'magenta', 'darkorange', 'blue', 'yellow'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC and AUC for %d classes' % (n_classes))
        plt.legend(loc="lower right")
        plt.show()


	def metrics(self):
		Y_test = self.model_adaptor.get_Y()
		y_pred_class = self.model_adaptor.get_pred_class()

		# accuracy
        self.accuracy = accuracy_score(y_true=Y_test, y_pred=y_pred_class)
        print("Accuracy: " + str(self.accuracy) + '\n')
        print("Classification report:\n")
        # print metrics
        print(classification_report(y_true=Y_test,
                                    y_pred=y_pred_class,
                                    target_names=list(map(lambda x: 'class %d' % x, self.classes))))
        # print confusion metrics
        print("Confusion metrics:\n")
        print(confusion_matrix(y_true=Y_test, y_pred=y_pred_class))
