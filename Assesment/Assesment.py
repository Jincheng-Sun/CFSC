from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keras import models
from scipy import interp
from sklearn.metrics import roc_curve, auc, confusion_matrix,f1_score,\
    classification_report,precision_score,recall_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
import os
#block Info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class Assesment_Module:
    def __init__(self, model, X_test, Y_test):
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test

    def _load_dataset(self,shape = None):
        self.X_test = np.reshape(self.X_test, shape)
        self.n_classes = np.max(self.Y_test)+1
        self.classes = np.arange(self.n_classes).tolist()
        self.Y_test_onehot = label_binarize(self.Y_test, classes=self.classes)

    def _predict(self):
            #   output: [1 2 1 4 3 ...]
            ##  type: ndarray | shape: [num,] exp:[4000,]
            #   For metrics
            self.y_pred_class = self.model.predict_classes(self.X_test)
            #   output: [[4.2632757e-05 8.5065442e-01 4.1671317e-02 2.6266620e-04 1.0736893e-01]
            #           ...
            #         [3.6251792e-03 6.7478888e-02 3.6363685e-01 4.1335770e-03 5.6112546e-01]]
            ##  type: ndarray | shape: [num,cls] exp:[4000,5]
            #   For RoC
            self.y_pred_score = self.model.predict(self.X_test)

    def _drawROC(self):
        # Plot RoC chart
        lw = 1
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.Y_test_onehot[:, i], self.y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(self.Y_test_onehot.ravel(), self.y_pred_score.ravel())
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

    def _metrics(self):
        # print metrics
        self.accuracy = accuracy_score(y_true=self.Y_test, y_pred=self.y_pred_class)
        print("Accuracy: "+str(self.accuracy)+'\n')
        print("Classification report:\n")
        print(classification_report(y_true=self.Y_test,
                                    y_pred=self.y_pred_class,
                                    target_names=list(map(lambda x:'class %d'%x,self.classes))))
        print("Confusion metrics:\n")
        print(confusion_matrix(y_true=self.Y_test, y_pred=self.y_pred_class))



def assesment(model_type, model_path, testX_path, testY_path, testX_shape):
    #   model_type: keras, sklearn...
    #   model_path: path to your model
    #   testX/Y_path: path of testset, type: .npy
    #   testX_shape: shape of input data X
    if model_type == 'keras':
        model = models.load_model(model_path)
    else:pass

    #
    X_test = np.load(testX_path)
    Y_test = np.load(testY_path)
    keras_assesment = Assesment_Module(model, X_test, Y_test)
    keras_assesment._load_dataset(shape=testX_shape)
    keras_assesment._predict()
    keras_assesment._drawROC()
    keras_assesment._metrics()

assesment(model_type='keras',
          model_path='../models/80000NN',
          testX_path='../data/test_x.npy',
          testY_path= '../data/test_y.npy',
          testX_shape=[4000,10000])



