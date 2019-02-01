# # import pandas as pd
# # import numpy as np
# # from keras import models, Sequential
# # from keras.layers import BatchNormalization, Conv2D, Dense, Activation, Flatten
# # from sklearn.model_selection import train_test_split
# # from sklearn import preprocessing
# # from sklearn.metrics import accuracy_score
# # from keras.utils.np_utils import to_categorical
# # from sklearn.model_selection import StratifiedKFold
# # from keras.utils import np_utils
# #
# #
# # model = Sequential()
# # model.add(Conv2D(256, input_shape=[100,100,1],kernel_size=[2, 100], strides=[1,100], padding='same'))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# # model.add(Flatten())
# # model.add(Dense(256))
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))
# # model.add(Dense(5, activation='softmax'))
# # model.compile(loss='categorical_crossentropy',
# #               optimizer='adam',
# #               metrics=['accuracy'])
# # model.summary()
# #
# # file1 = '../data/80000_trainset.csv'
# # file2 = '../models/CBOW.model'
# # file3 = '../data/train_x.npy'
# # file4 = '../data/train_y.npy'
# # file5 = '../data/4000_testset.csv'
# # file6 = '../data/test_x.npy'
# # file7 = '../data/test_y.npy'
# #
# # file8 = '../data/labels.txt'
# # file8_all = '../data/all_labels.txt'
# #
# # file9 = '../data/train_x_ex.npy'
# # file10 = '../data/train_y_ex.npy'
# # file11 = '../data/test_x_ex.npy'
# # file12 = '../data/test_y_ex.npy'
# # # def label_dic():
# # def conv_label(**kwargs):
# #     labels_file = kwargs['labels_file']
# #     all_labels_file = kwargs['all_labels']
# #     labels = {}
# #     all_labels = {}
# #     file = open(labels_file, 'r', encoding='gb18030')
# #     for line in file:
# #         labels[line.split(',')[0]] = int(line.split(',')[1])
# #     file_all = open(all_labels_file, 'r', encoding='gb18030')
# #     for line in file_all:
# #         all_labels[line.split(',')[0]] = int(line.split(',')[1])
# #     list = []
# #     for line in labels:
# #         list.append([labels[line], all_labels[line], line])
# #     list.sort()
# #     key = 0
# #     x = 1
# #     for i in range(len(list)):
# #         if list[i][0] == key:
# #             list[i][1] = x
# #             x += 1
# #         else:
# #             key += 1
# #             x = 1
# #             list[i][1] = x
# #             x += 1
# #
# #     dict = {list[i][2]: [list[i][0], list[i][1]] for i in range(len(list))}
# #     return dict
# #
# # d=conv_label(labels_file=file8,all_labels=file8_all)
#
# from Training.NN_training_adaptor import NN_training_adaptor
#
# nadpt = NN_training_adaptor
# nadpt.load_data(nadpt,'../data/train_x.npy','../data/train_y.npy','../data/test_x.npy','../data/test_y.npy')
# x, y = nadpt.label_data(nadpt,layer=1,expand=0,Y_data=nadpt.Y_train)
# a,b,c,d,e=nadpt.create_dataset(nadpt,Y_data=x,label_num=y)
a = [0,0,0,1,1,1,2,3,4,4,4,4]
b = [1,2,3,4,5,6,7,8,9,1,2,3]
aa=[]
bb=[]
for x,y in zip(a,b):
    if x!=1:
        aa.append(x)
        bb.append(y)

