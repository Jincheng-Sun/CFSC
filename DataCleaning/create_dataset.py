from DataCleaning import raw2valid, count_dept, valid2dataset, splitData
from Clustering.Optics_cluster import create_label


def raw2dataset():
    pass


valid_data = '/Users/sunjincheng/Documents/valid_data_all.csv'
all_label = '../data/all_labels.txt'
label = '../data/labels.txt'
dataset_file = '../data/trainset.csv'
train_set = '../data/trainset.csv'
test_set = '../data/testset.csv'


# Each time create dataset from valid data
# will rerun clustering, thus labels change
def valid2set(in_file, per):

    lines, count = count_dept.count_dept(in_file, all_label)
    create_label(all_label, count, label)
    valid2dataset.valid2set(label, in_file, dataset_file)
    splitData.split(dataset_file, int(lines * per), train_set, test_set)

def extract_data(dataset_file,num,train = True):
    if train:
        out_file = '../data/%d_trainset.csv'%(num)
    else:
        out_file = '../data/%d_testset.csv'%(num)
    splitData.extract(dataset_file,num,out_file)

valid2set(valid_data,0.8)
extract_data(train_set,80000,True)
extract_data(test_set,4000,False)