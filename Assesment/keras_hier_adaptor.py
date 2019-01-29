from Assesment.model_keras_adaptor import ModelKerasAdaptor
from Assesment.hierarchical_adaptor import Hierarchical_Adaptor
import numpy as np


class Keras_hier_adaptor(Hierarchical_Adaptor):
    def __init__(self):
        super().__init__()

    def init_params(self, model_type, network_size):
        # model_type: model type                        Type: String    exp: '.h5py'
        # network_size: how many models for each layer  Type: List      exp: [1,5]
        # layer: how many layers                        Type: Int       exp: 2
        self.model_type = model_type
        self.network_size = network_size
        self.layers = len(network_size)

    def load_data(self, x_file_path, y_file_path, models_path):
        # x(y)_file_path: test data path                Type: String
        # models_path: list of model paths              Type: String List   exp:['aaa.h5py','bbb.h5py'...]
        # Y_data label changed into label path          exp:[124,6,65,33]-->[[4,124],[1,6],[3,65],[2,33]]
        # for the first item [4,124], 4 is predict label for the 1st layer and 124 the 2nd layer.
        self.X_data = np.load(x_file_path)
        self.Y_data = np.load(y_file_path)
        self.models_path = models_path

    def build_network(self):
        # instantiate models for each layer, return a list of models
        models_all = []
        for l in range(self.layers):
            models = []
            for i in range(self.network_size[l]):
                model_path = self.models_path.pop(0)
                model = ModelKerasAdaptor()
                model.load(model_path)
                models.append((l, i, model))
            models_all.append(models)
        return models_all

    def fit_data(self, models_all):
        # Fit data ONE BY ONE, first goes through 1st layer, get predicted label and then decide
        # which is next model to fit data in
        # Returns list of predicted label path          exp:[[4,121],[1,6],[2,35],[2,33]...]
        predicts = []
        for data in self.X_data:
            # only for full tree of models
            model_index = 0
            predict = []
            for l in self.layers:
                model = models_all[model_index]
                l_pred = model.predict_classes(data)
                model_index = l_pred
                predict.append(l_pred)
            predicts.append(predict)
        return predicts

    def fit_data2(self, models_all):
        # Fit ALL data into ALL models, and returns list of predicted label path
        # Return                                       exp:[[4,121],[1,6],[2,35],[2,33]...]
        predicts = []
        pred = []
        for models in models_all:
            if models == 0:
                pred_this_layer = models.predict_classes(self.X_data)

            else:
                for model in models:
                    pred.append(model.predict_classes(self.X_data))

                for i in range(len(pred_this_layer)):
                    pred_this_layer[i] = pred[pred_this_layer[i]][i]

            predicts.append(pred_this_layer)
            pred = []

        pred_list = []
        for i in range(len(predicts[0])):
            predict = []
            for j in range(len(predicts)):
                predict.append(predicts[j][i])
            pred_list.append(predict)

        return pred_list

    def hierarchical_acc(self, pred_list, real_list, n = 1):
        # pred_list: list of predicted label path       exp:[[4,121],[1,6],[2,36],[2,33]...]
        # real_list: list of real label path            exp:[[4,124],[1,6],[3,65],[2,33]...]
        # Returns hP, hR, F-n score                     dtype: float
        INTER = []
        PRED = []
        REAL = []
        for pred, real in zip(pred_list, real_list):
            inter = [i for i in pred if i in real]
            INTER.append(len(inter))
            PRED.append(len(pred))
            REAL.append(len(real))
        INTER = float(sum(INTER))
        PRED = float(sum(PRED))
        REAL = float(sum(REAL))
        hP = INTER / PRED
        hR = INTER / REAL
        Fn = ((n * n + 1) * hP * hR) / (n * n * hP + hR)
        return hP, hR, Fn


