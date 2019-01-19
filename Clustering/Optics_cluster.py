from sklearn.cluster import DBSCAN, KMeans
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import jieba
import numpy as np
import matplotlib.pyplot as plt

model = Word2Vec.load('../models/CBOW.model')
file = '../data/all_labels.txt'

dimention = 2


def avg(num,label_file):
    dataset = np.array([])
    labels = {}
    for line in open(label_file, 'r', encoding='utf-8'):
        employer = line.split(',')[0]
        words = jieba.cut(employer)

        average = np.zeros(100)
        count = 0;
        for word in words:
            try:
                average += model[word]
                count += 1
            except:
                pass

        if count != 0:
            avg = average / count
            dataset = np.append(dataset, avg)
            labels[employer] = avg
        else:
            dataset = np.append(dataset, average)
            labels[employer] = average

    dataset = dataset.reshape(num, 100)
    return dataset, labels


# dataset: [num,100]
# labels: { employer : vector }
# dataset, labels = avg()
#
# pca = PCA(dimention)
# dataset = pca.fit_transform(dataset)
# clust = KMeans(n_clusters=5)
# opt = clust.fit(dataset)
# center = opt.cluster_centers_
# opt = KMeans(n_clusters=5, algorithm='full', init=center).fit(dataset)
#
# # gmm = GaussianMixture(n_components=5)
# # opt = gmm.fit(dataset)
#
# class0 = np.array([])
# class1 = np.array([])
# class2 = np.array([])
# class3 = np.array([])
# class4 = np.array([])
# for data in dataset:
#     data = np.reshape(data, [1, dimention])
#     pred = opt.predict(data)
#     if pred == 0:
#         class0 = np.append(class0, data)
#     if pred == 1:
#         class1 = np.append(class1, data)
#     if pred == 2:
#         class2 = np.append(class2, data)
#     if pred == 3:
#         class3 = np.append(class3, data)
#     if pred == 4:
#         class4 = np.append(class4, data)
#
# class0 = class0.reshape(-1, dimention)
# class1 = class1.reshape(-1, dimention)
# class2 = class2.reshape(-1, dimention)
# class3 = class3.reshape(-1, dimention)
# class4 = class4.reshape(-1, dimention)
#
#
def plot2d(dataset, name):
    plt.scatter(dataset[:, 0], dataset[:, 1], label=name)


def plotall():
    # pca = PCA(2)
    # newdata = pca.fit_transform(class0)
    # plt.scatter(newdata[:, 0], newdata[:, 1], label='类0')
    plot2d(class0, 'class 0')
    plot2d(class1, 'class 1')
    plot2d(class2, 'class 2')
    plot2d(class3, 'class 3')
    plot2d(class4, 'class 4')
    plt.legend(loc='upper right')
    plt.show()


def plot3d(dataset, name, ax):
    ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], label=name)


def plotall3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot3d(class0, 'class 0', ax)
    plot3d(class1, 'class 1', ax)
    plot3d(class2, 'class 2', ax)
    plot3d(class3, 'class 3', ax)
    plot3d(class4, 'class 4', ax)
    plt.show()


def save_cls():
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []

    for line in open(file, 'r', encoding='utf-8'):
        employer = line.split(',')[0]
        vector = labels[employer]
        vector = vector.reshape(1, 100)
        vector = pca.transform(vector)
        Y = opt.predict(vector)
        if Y == 0:
            cluster0.append(employer)
        if Y == 1:
            cluster1.append(employer)
        if Y == 2:
            cluster2.append(employer)
        if Y == 3:
            cluster3.append(employer)
        if Y == 4:
            cluster4.append(employer)

    with open('cluster.txt', 'w', encoding='gb18030') as f:
        f.write('第0类' + '\n')
        f.write(str(cluster0))
        f.write('\n')
        f.write('第1类' + '\n')
        f.write(str(cluster1))
        f.write('\n')
        f.write('第2类' + '\n')
        f.write(str(cluster2))
        f.write('\n')
        f.write('第3类' + '\n')
        f.write(str(cluster3))
        f.write('\n')
        f.write('第4类' + '\n')
        f.write(str(cluster4))
    with open('labels.txt', 'a+', encoding='gb18030') as f:
        for cluster in cluster0:
            f.write(cluster + ',0' + '\n')
        for cluster in cluster1:
            f.write(cluster + ',1' + '\n')
        for cluster in cluster2:
            f.write(cluster + ',2' + '\n')

        for cluster in cluster3:
            f.write(cluster + ',3' + '\n')
        for cluster in cluster4:
            f.write(cluster + ',4' + '\n')


def create_label(in_file,num,out_label):
    dataset, labels = avg(num,in_file)
    pca = PCA(dimention)
    dataset = pca.fit_transform(dataset)
    clust = KMeans(n_clusters=5)
    opt = clust.fit(dataset)
    center = opt.cluster_centers_
    opt = KMeans(n_clusters=5, algorithm='full', init=center).fit(dataset)
    # gmm = GaussianMixture(n_components=5)
    # opt = gmm.fit(dataset)
    class0 = np.array([])
    class1 = np.array([])
    class2 = np.array([])
    class3 = np.array([])
    class4 = np.array([])
    for data in dataset:
        data = np.reshape(data, [1, dimention])
        pred = opt.predict(data)
        if pred == 0:
            class0 = np.append(class0, data)
        if pred == 1:
            class1 = np.append(class1, data)
        if pred == 2:
            class2 = np.append(class2, data)
        if pred == 3:
            class3 = np.append(class3, data)
        if pred == 4:
            class4 = np.append(class4, data)

    class0 = class0.reshape(-1, dimention)
    class1 = class1.reshape(-1, dimention)
    class2 = class2.reshape(-1, dimention)
    class3 = class3.reshape(-1, dimention)
    class4 = class4.reshape(-1, dimention)
    plot2d(class0, 'class 0')
    plot2d(class1, 'class 1')
    plot2d(class2, 'class 2')
    plot2d(class3, 'class 3')
    plot2d(class4, 'class 4')
    plt.legend(loc='upper right')
    plt.show()

    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []

    for line in open(in_file, 'r', encoding='utf-8'):
        employer = line.split(',')[0]
        vector = labels[employer]
        vector = vector.reshape(1, 100)
        vector = pca.transform(vector)
        Y = opt.predict(vector)
        if Y == 0:
            cluster0.append(employer)
        if Y == 1:
            cluster1.append(employer)
        if Y == 2:
            cluster2.append(employer)
        if Y == 3:
            cluster3.append(employer)
        if Y == 4:
            cluster4.append(employer)

    with open('cluster.txt', 'w', encoding='gb18030') as f:
        f.write('第0类' + '\n')
        f.write(str(cluster0))
        f.write('\n')
        f.write('第1类' + '\n')
        f.write(str(cluster1))
        f.write('\n')
        f.write('第2类' + '\n')
        f.write(str(cluster2))
        f.write('\n')
        f.write('第3类' + '\n')
        f.write(str(cluster3))
        f.write('\n')
        f.write('第4类' + '\n')
        f.write(str(cluster4))
    with open(out_label, 'a+', encoding='gb18030') as f:
        for cluster in cluster0:
            f.write(cluster + ',0' + '\n')
        for cluster in cluster1:
            f.write(cluster + ',1' + '\n')
        for cluster in cluster2:
            f.write(cluster + ',2' + '\n')

        for cluster in cluster3:
            f.write(cluster + ',3' + '\n')
        for cluster in cluster4:
            f.write(cluster + ',4' + '\n')

# clust = DBSCAN(min_samples= 3, rejection_ratio=0.5)
# opt = clust.fit_predict(X)
# labels = clust.labels_
# print(max(labels))
# optics_dict = defaultdict(list)
# for i,v in enumerate(labels):
#     optics_dict[v].append(industry_lst[i])
#
# print('min_samples= 3, rejection_ratio=0.5')
# print(sorted(optics_dict.items(), key= lambda k:k[0]))
