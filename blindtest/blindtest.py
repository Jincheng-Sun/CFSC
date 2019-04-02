from DataCleaning.cleaning import hgdProcess_dept
from Training.create_dataset import create
import pandas as pd

# rawdata = pd.read_excel('../data/blindtest.xlsx')
# invalid_list = ['其他单位', '省外单位','nan' , '省级单位','除海口外的市县','无效归属','无效数据','政府单位']
# inv_data = rawdata.loc[rawdata['主办单位'].isin(invalid_list)]
# rawdata = rawdata.drop(inv_data.index)
# rawdata = rawdata.dropna(how='any')
#
# rawdata['主办单位'].value_counts()
#
# rawdata['处置单位']=rawdata['主办单位'].apply(hgdProcess_dept)
#
# rawdata.to_csv('../data/blindtestset.csv',encoding='gb18030')

create(input = '../data/blindtestset.csv',output1='../data/blind_x.npy',output2='../data/blind_y.npy',stopword=True,is_expanded=True)


from keras.models import load_model
import numpy as np
from sklearn.metrics import log_loss,accuracy_score,classification_report
model = load_model('../models/80000NN.h5py')

x_test = np.load('../data/blind_x.npy')
y_test = np.load('../data/blind_y.npy')

y_pred = model.predict(x_test)

ll = log_loss(y_test,y_pred,labels=np.arange(157))

acc = accuracy_score(y_test,np.argmax(y_pred,axis=1))
result = classification_report(y_test,np.argmax(y_pred,axis=1),digits=4)