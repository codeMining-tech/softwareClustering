
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import scipy
from scipy.special._ufuncs import kl_div

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets.samples_generator import make_swiss_roll


from scipy.spatial.distance import pdist
def getDistance(P,Q):
    BC = np.sum(np.sqrt(np.array(P)*np.array(Q)))  
    h = np.sqrt(1 - BC+0.00000001) 
    b = -np.log(BC)
    X = np.vstack([P, Q])
    d_C = 1 - pdist(X, 'cosine')
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
    ward = AgglomerativeClustering(affinity='precomputed', n_clusters=clustersNum, linkage='average').fit(d_mat)
    label = ward.labels_ 
    return label
  