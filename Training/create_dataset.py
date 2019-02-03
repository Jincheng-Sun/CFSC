import csv
import sys
sys.path.append('/home/oem/Projects/CFSC/')
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from DataCleaning import cleaning as cl
from collections import Counter
file1 = '../data/80000_trainset.csv'
file2 = '../models/CBOW.model'
file3 = '../data/train_x.npy'
file4 = '../data/train_y.npy'
file5 = '../data/4000_testset.csv'
file6 = '../data/test_x.npy'
file7 = '../data/test_y.npy'

file8 = '../data/labels.txt'
file8_all = '../data/all_labels.txt'

file9 = '../data/train_x_ex.npy'
file10 = '../data/train_y_ex.npy'
file11 = '../data/test_x_ex.npy'
file12 = '../data/test_y_ex.npy'
# def label_dic():
def conv_label(is_expanded, **kwargs):
    labels_file = kwargs['labels_file']
    all_labels_file = kwargs['all_labels']
    labels = {}
    all_labels = {}
    file = open(labels_file, 'r', encoding='gb18030')
    for line in file:
        labels[line.split(',')[0]] = int(line.split(',')[1])
    file_all = open(all_labels_file, 'r', encoding='gb18030')
    for line in file_all:
        all_labels[line.split(',')[0]] = int(line.split(',')[1])
    list = []
    for line in labels:
        list.append([labels[line], all_labels[line], line])
    list.sort()
    key = 0
    x = 0
    for i in range(len(list)):
        if list[i][0] == key:
            list[i][1] = x
            x += 1
        else:
            key += 1
            x = 0
            list[i][1] = x
            x += 1

    dict = {list[i][2]: [list[i][0], list[i][1]] for i in range(len(list))}
    if is_expanded:
        return all_labels
    else:
        return dict
# for hier models
# labels = conv_label(labels_file=file8,all_labels=file8_all)
# for expanded labels

def stopwords(file):
    stopwords = [line[0:-1] for line in open(file, 'r', encoding='utf-8').readlines()]
    return set(stopwords)

def create(input,output1,output2,stopword = False,is_expanded = False):
    labels = conv_label(labels_file=file8, all_labels=file8_all, is_expanded=is_expanded)
    if stopword:
        stopword_list = stopwords('../data/baidu+chuanda.txt')
    else:
        stopword_list = []
    labels_trans = cl.further_clean()

    #input: raw data
    #output1: train(test)set x, shape = [40000,10000]
    #output2: train(test)set y, shape = [40000,1]
    model = Word2Vec.load(file2)
    file = open(input, 'r', encoding='gb18030')
    lines = csv.reader(file)
    count = 0
    dataset_x = []
    dataset_y = []
    for line in lines:
        count += 1
        if (count == 1):
            continue

        # if (count == 40002):
        #     break


        # if(count<80000):
        #     continue
        # if (count == 80000):
        #     break

        if (count % 1000 == 0):
            print(count)
        id = line[0]
        sent = line[-4]
        label = line[-1]
        employer = line[-2]
        try:
            employer = labels_trans[employer]
            employer = re.sub('\n', '', employer)
        except:
            pass

        sent = re.sub('市民来电咨询', '', sent)
        sent = re.sub('市民来电反映', '', sent)
        sent = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[a-zA-Z0-9+——！，。？、~@#￥%……&*（）《》：:]+", "", sent)
        splits = jieba.cut(sent)

        vector = np.array([])
        for word in splits:

            if word in stopword_list:
                continue
            try:
                vec = model[word]
                vector = np.append(vector, vec)
                if (vector.shape[0] == 10000):
                    break
            except Exception:
                continue

        if (vector.shape[0] < 10000):
            pendding = np.zeros(10000 - vector.shape[0])
            vector = np.append(vector, pendding)
            # vector = vector.tolist()
        dataset_x.append(vector)
        dataset_y.append(labels[employer])
        # print(employer)
        # print(labels[employer])
    np.save(output1, dataset_x)
    np.save(output2, dataset_y)
#
# label_dic()

#
create(file5,file6,file7,stopword=True,is_expanded=False)
# create(file1,file3,file4,stopword=True)


