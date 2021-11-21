
"""

    @date 4-17-2021
    @detail 实现一个层次聚类算法：
    @input 一个文件实体列表、可以计算其
"""

# 层次聚类
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy
from scipy.special._ufuncs import kl_div

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets.samples_generator import make_swiss_roll

# n_samples = 1500
# noise = 0.05
# X, _ = make_swiss_roll(n_samples, noise)#卷型数据集
# #进行放缩
# X[:, 1] *= .5

#ward会利用AgglomerativeClustering对象，尝试平方和最小化所有集群内的差异

from scipy.spatial.distance import pdist
def getDistance(P,Q):
    BC = np.sum(np.sqrt(np.array(P)*np.array(Q)))

    # Hellinger距离：
    h = np.sqrt(1 - BC+0.00000001)

    # 巴氏距离： 交叉熵
    b = -np.log(BC)
    #巴氏距离明显优于
    # 散度 scipy.stats.entropy(P, Q)







    # 方法二：根据scipy库求解
    # X = np.vstack([P,Q])
    # d_j = pdist(X, 'jaccard')
    # #return float('%.8g'%b)
    #
    # # 余弦距离
    X = np.vstack([P, Q])
    d_C = 1 - pdist(X, 'cosine')
    #
    # X = np.vstack([P, ])
    # d_O = pdist(X)
    return b


def getDistanceMat(LDAVecMat):
    distanceMat = np.zeros((len(LDAVecMat),len(LDAVecMat)))
    for i in range(0,len(LDAVecMat)):
        for j in range(0,len(LDAVecMat)):
            if i == j:
                distanceMat[i][j] = 0
            else:
                distanceMat[i][j] = getDistance(np.array(LDAVecMat[i]),np.array(LDAVecMat[j]))
            if(distanceMat[i][j]<0):
                distanceMat[i][j] = 0

    return distanceMat
def H_AgglomerativeClustering(LDAVecMat,clustersNum =13):
    d_mat = getDistanceMat(LDAVecMat)
    # print(d_mat)
    # print(type(d_mat))
    ward = AgglomerativeClustering(affinity='precomputed', n_clusters=clustersNum, linkage='average').fit(d_mat)
    label = ward.labels_  # 得到lable值
    return label
#ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
# label =H_AgglomerativeClustering(X)
#
# fig = plt.figure()
# ax = p3.Axes3D(fig)
# ax.view_init(7, -80)
# for l in np.unique(label):
#     ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
#                color=plt.cm.jet(np.float(l) / np.max(label + 1)),
#                s=20, edgecolor='k')
# plt.show()

###
def kMeansCluster(data,k):
    #final = open('c:/test/final.dat' , 'r')
    #二维列表、
    #data = [line.strip().split('\t') for line in final]
    #feature = [[float(x) for x in row[3:]] for row in data]
    feature = data
    #调用kmeans类   # 默认的距离公式是 欧氏距离
    clf = KMeans(n_clusters=k)
    s = clf.fit(feature)
    # print(s)
    # 9个中心
    # print(clf.cluster_centers_)

    # 每个样本所属的簇
    # print(clf.labels_)

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # print(clf.inertia_)

    # 进行预测
    # print(clf.predict(feature))

    # 保存模型

    return clf
# K-Means明显比不上 层次聚类
def K_KMeansClustering(LDAVecMat,clustersNum =13):
    clf = KMeans(n_clusters=clustersNum)
    s = clf.fit(LDAVecMat)
    return clf.predict(LDAVecMat)
        # 进行预测
        # print(clf.predict([entity.ldaTurplesFeature for entity in EntityList]))
        # print(len(clf.predict([entity.ldaTurplesFeature for entity in EntityList])))

# P = [0.2,0.8]
# Q = [0.3,0.7]
# C = [0.9,0.1]
# i = getDistance(P,Q)
# j = getDistance(P,C)
# m = getDistance(P,P)
# print(i)
# print(j)
# print(m)
p1 = [0.6,0.2,0.1,0.1]
p2 = [0.1,0.1,0.1,0.7]
p3 = [0.7,0.1,0.1,0.1]
p4 = [0.6,0.1,0.2,0.1]
p5 = [0.5,0.2,0.2,0.1]
r = getDistanceMat([p1,p2,p3,p4,p5])
