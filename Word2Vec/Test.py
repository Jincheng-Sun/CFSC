from gensim.models import Word2Vec

file1 = '../models/CBOW.model'
model = Word2Vec.load(file1)
# res = model.most_similar('问题')
print(model[u'公安局'])