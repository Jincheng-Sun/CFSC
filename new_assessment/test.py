from new_assessment.assess_model import AssessModel
from new_assessment.keras_model_adaptor import KerasModelAdaptor
import numpy as np

K = KerasModelAdaptor('../models/NN00','../data/test_y.npy','../data/test_x.npy',[-1,10000])
classes = list(range(5))
A = AssessModel(K,classes)
A.draw_roc()
A.metrics()
t = np.load('../data/test_y.npy')
origin_label = np.load('../data/origin_label.npy')
pred_class = A.pred_class
Y_data = A.Y_data
ori_label0 = []
ori_label1 = []
ori_label2 = []
ori_label3 = []
ori_label4 = []
for origin,cls,predcls in zip(origin_label,Y_data,pred_class):
    if predcls == 0:
        if cls == 0:
            ori_label0.append(origin)
        elif cls == 1:
            ori_label1.append(origin)
        elif cls == 2:
            ori_label2.append(origin)
        elif cls == 3:
            ori_label3.append(origin)
        elif cls == 4:
            ori_label4.append(origin)
from collections import Counter
o0 = Counter(ori_label0)
o1 = Counter(ori_label1)
o2 = Counter(ori_label2)
o3 = Counter(ori_label3)
o4 = Counter(ori_label4)

with open('../data/misjudge_label0.txt','w',encoding='gb18030') as f:
    f.write('统计格式：{原分类：判断到0类的数量}\n')
    f.write('统计数量：4000条中判断为class0的数据\n')
    f.write('第0类正确判断到第0类 数量' + str(len(ori_label0)) + '\n')
    f.write(str(o0) + '\n')
    f.write('第1类误判到第0类 数量' + str(len(ori_label1))+'\n')
    f.write(str(o1)+'\n')
    f.write('第2类误判到第0类 数量' + str(len(ori_label2))+'\n')
    f.write(str(o2)+'\n')
    f.write('第3类误判到第0类 数量' + str(len(ori_label3))+'\n')
    f.write(str(o3)+'\n')
    f.write('第4类误判到第0类 数量' + str(len(ori_label4))+'\n')
    f.write(str(o4)+'\n')
# from new_assessment.hier_assess_model import HierAssessModel
# from new_assessment.keras_hier_model_adaptor import KerasHierModelAdaptor
#
# models = ['../models/NN00', '../models/NN10', '../models/NN11', '../models/NN12', '../models/NN13', '../models/NN14']
# K = KerasHierModelAdaptor(models_path=models,
#                           x_file_path='../data/test_x.npy',
#                           y_file_path='../data/test_y.npy',
#                           network_size = [1,5],input_shape=[-1,10000])
# K.build_network()
# A = HierAssessModel(K)
# A.hierarchical_assessment(1)