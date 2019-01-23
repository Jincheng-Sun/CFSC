# from gensim.models import Word2Vec
def stopwords(file):
    stopwords = [line[0:-1] for line in open(file, 'r', encoding='utf-8').readlines()]
    return set(stopwords)

a = stopwords('../data/baidu+chuanda.txt')
#
# file1 = '../models/CBOW.model'
# model = Word2Vec.load(file1)
# # res = model.most_similar('问题')
# print(model[u'公安局'])