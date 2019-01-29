from Assesment.model_adaptor import IModelAdaptor
import json
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from keras import models
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
import os

# block Info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class ModelKerasAdaptor(IModelAdaptor):
    def __init__(self):
        super().__init__()

    def load(self, model_file_path):
        # load model
        self.classifier_model = models.load_model(model_file_path)

    def predict_score(self, in_obj):
        # classification score
        return self.classifier_model.predict(in_obj)

    def predict_classes(self, in_obj):
        # predict classes
        return self.classifier_model.predict_classes(in_obj)

    def file_postfix(self):
        # return model type
        return ".h5py"

    def draw_RoC(self, **kwargs):
        # take inputs of Y_test and pred_score
        Y_test = kwargs['Y_test']
        y_pred_score = kwargs['pred_score']
        # count classes
        n_classes = np.max(Y_test) + 1
        classes = np.arange(n_classes).tolist()
        # turn Y_test into one-hot format
        Y_test_onehot = label_binarize(Y_test, classes=classes)

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

    def metrics(self, **kwargs):
        # take inputs of Y_test and pred_class to calculate metrics
        Y_test = kwargs['Y_test']
        y_pred_class = kwargs['pred_class']
        n_classes = np.max(Y_test) + 1
        classes = np.arange(n_classes).tolist()
        # accuracy
        self.accuracy = accuracy_score(y_true=Y_test, y_pred=y_pred_class)
        print("Accuracy: " + str(self.accuracy) + '\n')
        print("Classification report:\n")
        # print metrics
        print(classification_report(y_true=Y_test,
                                    y_pred=y_pred_class,
                                    target_names=list(map(lambda x: 'class %d' % x, classes))))
        # print confusion metrics
        print("Confusion metrics:\n")
        print(confusion_matrix(y_true=Y_test, y_pred=y_pred_class))

    def convert_data(self,dict,shape):
        # take dictionary and output ndarray
        X_test = []
        Y_test = []
        for key in dict:
            X_test.append(key)
            Y_test.append(dict[key])
        return np.array(X_test).reshape(shape=shape), np.array(Y_test)

    def serialize(self, out_obj):
        return json.dumps(out_obj.tolist())

    def deserialize(self, in_str = None, **kwargs):
        # shape = kwargs['shape']
        # return self.convert_data(json.loads(in_str),shape=shape)

        # return reshaped X_test, origin Y_test

        return np.reshape(np.load(kwargs['x_filepath']), newshape=kwargs['shape']), np.load(kwargs['y_filepath'])
