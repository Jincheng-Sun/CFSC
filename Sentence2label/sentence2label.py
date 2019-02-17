import argparse
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from keras import models

def stopwords(file):
    stopwords = [line[0:-1] for line in open(file, 'r', encoding='utf-8').readlines()]
    return set(stopwords)

def convert(sentence, stopfile,w2v_model_path):

    stopword_list = stopwords(stopfile)

    model = Word2Vec.load(w2v_model_path)
    sent = sentence
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
        return vector.reshape([1,10000])
def index2label(index,label_file):
    file = open(label_file, 'r', encoding='gb18030')
    i2l = {}
    for line in file:
        i2l[int(line.split(',')[1])] = line.split(',')[0]
    return i2l[index[0]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',type=str)
    parser.add_argument('-w2v_path', type=str)
    parser.add_argument('-stopword', type=str)
    parser.add_argument('-model_path', type=str)
    parser.add_argument('-label_file', type=str)

    args = parser.parse_args()


    sentence = args.s
    w2v_model_path = args.w2v_path
    stopword_path = args.stopword
    model_path = args.model_path
    label_file = args.label_file


    vector = convert(sentence=sentence,stopfile=stopword_path,w2v_model_path=w2v_model_path)
    model = models.load_model(model_path)
    index = model.predict_classes(vector)
    label = index2label(index,label_file)
    print(label)

if __name__ == '__main__':
    label = main()

'python3 sentence2label.py -s 市民来电反映:其是龙华区滨濂村北社区3里138号海瑞学校后面居民，刚刚因有人拿刀到其家内将门砸坏，破坏楼顶结构，报警到海垦派出所，其刚从派出所回到家发现闹事的人还在，请公安局核实处理，谢谢！（请职能局按规定在30分钟内联系市民，响应处置） -w2v_path models/CBOW.model -stopword data/baidu+chuanda.txt -model_path models/80000NN.h5py -label_file data/all_labels.txt'