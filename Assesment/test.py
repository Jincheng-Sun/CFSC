from Assesment.model_keras_adaptor import ModelKerasAdaptor
from Assesment.keras_hier_adaptor import Keras_hier_adaptor
import numpy as np
kerasmodel = ModelKerasAdaptor()
kerasmodel.load('../models/80000NN.h5py')
#
model_type = kerasmodel.file_postfix()

X_test, Y_test = kerasmodel.deserialize(x_filepath='../data/test_x.npy',
                           y_filepath='../data/test_y.npy',
                           shape=[4000,10000])

pred_score = kerasmodel.predict_score(in_obj=X_test)
pred_class = kerasmodel.predict_classes(in_obj=X_test)

kerasmodel.draw_RoC(Y_test=Y_test,pred_score=pred_score)
kerasmodel.metrics(Y_test=Y_test,pred_class=pred_class)
#
# json = kerasmodel.serialize(pred_class)

# import json
#
# jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}';
#
# text = json.loads(jsonData)
#
# x,y=kerasmodel.convert_data(text)
#
# list1 = np.array([1,2,3,1,2,1,3,2,4,0])
# list2 = np.array([1,2,1,2,2,1,4,2,2,0])
#
# a=list1.__eq__(list2)*1


def assesment(model_adaptor):
    model_type = model_adaptor.file_postfix()
    if model_type == ".h5py":
        pass
    elif model_type == "joblib":
        pass

#
# a = [[0, 1, 4], [5, 1, 2], [2, 3, 1], [1, 2, 1]]
# b = [[0, 2, 4], [5, 1, 2], [2, 3, 1], [1, 1, 1]]
#
# K = Keras_hier_adaptor
# c = K.hierarchical_acc(K, pred_list=a, real_list=b, n=1)
# print(c)