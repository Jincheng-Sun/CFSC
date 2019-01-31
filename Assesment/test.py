# from Assesment.model_keras_adaptor import ModelKerasAdaptor
# from Assesment.keras_hier_adaptor import Keras_hier_adaptor
import numpy as np
import pandas as pd
# kerasmodel = ModelKerasAdaptor()
# kerasmodel.load('../models/80000NN.h5py')
# #
# model_type = kerasmodel.file_postfix()
#
# X_test, Y_test = kerasmodel.deserialize(x_filepath='../data/test_x.npy',
#                            y_filepath='../data/test_y.npy',
#                            shape=[4000,10000])
#
# pred_score = kerasmodel.predict_score(in_obj=X_test)
# pred_class = kerasmodel.predict_classes(in_obj=X_test)
#
# kerasmodel.draw_RoC(Y_test=Y_test,pred_score=pred_score)
# kerasmodel.metrics(Y_test=Y_test,pred_class=pred_class)
# #
# # json = kerasmodel.serialize(pred_class)
#
# # import json
# #
# # jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}';
# #
# # text = json.loads(jsonData)
# #
# # x,y=kerasmodel.convert_data(text)
# #
# # list1 = np.array([1,2,3,1,2,1,3,2,4,0])
# # list2 = np.array([1,2,1,2,2,1,4,2,2,0])
# #
# # a=list1.__eq__(list2)*1
#
#
# def assesment(model_adaptor):
#     model_type = model_adaptor.file_postfix()
#     if model_type == ".h5py":
#         pass
#     elif model_type == "joblib":
#         pass
#
# #
# # a = [[0, 1, 4], [5, 1, 2], [2, 3, 1], [1, 2, 1]]
# # b = [[0, 2, 4], [5, 1, 2], [2, 3, 1], [1, 1, 1]]
# #
# # K = Keras_hier_adaptor
# # c = K.hierarchical_acc(K, pred_list=a, real_list=b, n=1)
# # print(c)

from Training.NN_training_adaptor import NN_training_adaptor

nadaptor = NN_training_adaptor()
nadaptor.load_data('../data/train_x.npy','../data/train_y.npy','../data/test_x.npy','../data/test_y.npy')
label_list = [0,1,2,3,4]
path_list = ['../models/NN10','../models/NN11','../models/NN12','../models/NN13','../models/NN14']
for class_,path_ in zip(label_list,path_list):
    num_labels, x_train, y_train, x_val, y_val, Y_test = nadaptor.create_dataset(expand_class=class_)
    # model = nadaptor.network(cls_num=num_labels)
    # model = nadaptor.train_model(model=model, X_train=x_train, Y_train=y_train, x_val=x_val, y_val=y_val, save_path=path_)
    model = nadaptor.load_model('../models/NN10')
    nadaptor.assesment(model=model,X_test=nadaptor.X_test,Y_test=Y_test)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# from keras.utils.np_utils import to_categorical
# expand_class = 2
# Y = []
# Y_train = np.array([[0,0],[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7],[4,8],[4,9]])
# for fl, sl in zip(Y_train[:, 0], Y_train[:, 1]):
#     if fl == expand_class:
#         Y.append(sl+5)
#     else:
#         Y.append(fl)
# Y_train = np.array(Y)
# y_train = pd.DataFrame(Y_train)[0]
#
# # one-hot，5 category
# y_labels = list(y_train.value_counts().index)
# # y_labels = np.unique(y_train)
# le = preprocessing.LabelEncoder()
# le.fit(y_labels)
# num_labels = len(y_labels)
# y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
# pass
