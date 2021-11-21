import copy
import math

import networkx as nx
from Core.clusterArgorithm import getDistanceMat
from sklearn.neighbors import LocalOutlierFactor


def LOF(data, predict=None, k=25,fr = 0.2):
    result = []
    if 1==len(data):
        return result
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    clf = LocalOutlierFactor(n_neighbors=k, algorithm='auto', contamination=fr, n_jobs=-1,metric='precomputed')
    getDistanceMat(data)
    clf.fit(getDistanceMat(data))
    labels = clf._predict()
    #lofValues = clf._score_samples(data)

    for i in range(0,len(labels)):
        if -1 == labels[i]:
            result.append(i)
    return result

class OutLiersDetector():
    __outLiersNum__ = 1
    __outLiersNumRate__ = 0.2
    __TOPIC__ = 120 # default topic number
    __k__ = 20 # LOF paremeter K (the size of neighborhood)
    def __init__(self,EntList):
        self.m_EntList = EntList
        self.m_dirList = []
        return
    def getSumOfList(self,aList,anotherList):
        result = list()
        for (x,y) in zip(aList,anotherList):
            result.append(x+y)
        return result
    def selectOrpanNodesPost(self,theClustersWithNums,theTopicNum):
        result = []
        if 0 == self.__outLiersNum__:
            return result
        for cluster in theClustersWithNums:
            resultIndexList = LOF([self.m_EntList[i].LdaVector for i in cluster],None,
                               20,OutLiersDetector.__outLiersNumRate__)
            result.extend( [cluster[x] for x in resultIndexList])
        return  result


    def constructFeaturesSubset(self, theOrpanList):
            dirSet = set()
            for ent in self.m_EntList:
                dirSet.add(ent.dir)
            resultDict = dict()
            dirList = list(dirSet)
            self.m_dirList = dirList
            for i in range(0, len(dirSet)):
                resultDict[dirList[i]] = []
            for ent in self.m_EntList:
                if False == ent.notUse and ent.m_Id not in theOrpanList:
                    resultDict[ent.dir].append(ent.m_Id)
            result = [item[1] for item in list(resultDict.items())]
            return result

    def writeExpertPartitionGraph(self,CDG):
        expertP = self.constructClustersWithDir([])
        ECDG = copy.deepcopy(CDG)
        for i in range(0,len(expertP)):
            for entNum in expertP[i]:
                ECDG.nodes[entNum]['clusterID'] = i
                ECDG.nodes[entNum]['label'] = i
        nx.write_gexf(ECDG, 'CDG_Expert.gexf')
        return ECDG


    def selectOrpanNodesPre(self,theOrpanList):
        featuresSubset = self.constructFeaturesSubset(theOrpanList)  # construct features subset (first-level directory) feature ID
        return  self.selectOrpanNodesPost(featuresSubset,OutLiersDetector.__TOPIC__)


    def getDirNums(self,aCluster,theDir):
        result = 0
        for entNum in aCluster:
            if theDir == self.m_EntList[entNum].dir:
                result = result+1
        return result

