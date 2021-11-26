
import networkx as nx
from gensim import corpora, models, similarities
from Core.clusterArgorithm import H_AgglomerativeClustering
from Core.evaluator import Evaluator
from outliersAssignment import OutliersAssignment
from Core.outliersDetector import OutLiersDetector

case = 'content'
casePath = 'Cases/'+case
fileNameList = list()
dirLevel = 1   # first level dir
k = 13
class m_entity:
    _TYPE_ = 'FILE'
    def __init__(self,entityName,entityUniqueName):
        self.entityName = entityName
        self.entityUniqueName = entityUniqueName
        self.relName = ''
        self.entityDir=' ' # 存储实体·所在目录（文件夹）
        self.fileTyepe=''
        self.udbWordsDoc =list()
        self.normalWords = list()
        self.LdaVector = list()
        self.commentsWords = list()
        self.commentsNormalWords = list()
        self.commentsLdaVector = list()
        self.isMerge = False
        self.notUse = False
        self.m_Id =-1
        self.m_mFileId =-1
def attachDir2Ent(theEntList):
    for ent in theEntList:
        ent.dirList = ent.entityName.split('\\')[0:-1]
        ent.dir = ent.dirList[dirLevel]

MergedHFileNumList = []
def initFileEntity():
    FDG = nx.read_adjlist(casePath+'/FDG')

    fileInfo = open(casePath+'/filesInfo.txt')
    fileInfo_m = open(casePath+'/fileInfo_m.txt')
    entityList = list()
    for line in fileInfo.readlines():
        fileNameList.append(str(line).strip())
    for i in range(len(fileNameList)):
        entityNew = m_entity(fileNameList[i], fileNameList[i])
        entityNew.relativePathName = fileNameList[i]
        entityNew.fileTyepe = fileNameList[i].split('.')[-1]
        entityNew.m_Id = i
        entityList.append(entityNew)
    for line in fileInfo_m.readlines():
        mergePair = line.split('\t')
        entityList[fileNameList.index(mergePair[0].strip())].m_mFileId = fileNameList.index(mergePair[1].strip())
        entityList[fileNameList.index(mergePair[0].strip())].m_mFileId = fileNameList.index(mergePair[1].strip())
        entityList[fileNameList.index(mergePair[0].strip())].isMerge = True
        entityList[fileNameList.index(mergePair[1].strip())].notUse= True
        MergedHFileNumList.append(fileNameList.index(mergePair[1].strip()))

    attachDir2Ent(entityList)
    return entityList,FDG
def MergeFDGNode(CDG):
    MergedHFileNumSet = set(MergedHFileNumList)
    for ent in entityList:
        # = nx.DiGraph()
        if True == ent.isMerge and ent.m_mFileId in MergedHFileNumSet:

            for node in CDG.neighbors(str(ent.m_mFileId)):
                node = int(node)
                if int(node) == ent.m_Id:
                    break
                # if CDG.has_edge(int(node), ent.m_Id):
                #     CDG[node][ent.m_Id]['weight'] += CDG[node][ent.m_mFileId]['weight']
                # else:
                CDG.add_edge(str(node), str(ent.m_Id))
            CDG.remove_node(str(ent.m_mFileId))
            MergedHFileNumSet.remove(ent.m_mFileId)
    return CDG
def normalLdaTurples(turples,k):
    for i in range(0,k):
        if i not in [turple[0] for turple in turples]:
            turples.insert(i,(i,0))
    return [turple[1] for turple in turples]
def convertNum2Names(aPartitionWithNum,aSrcFileList):
    result =[]
    for cluster in aPartitionWithNum:
        result.append([aSrcFileList[num] for num in cluster])
    return  result


def getLDATopicMatrix(path):
    lda_multi = models.ldamulticore.LdaMulticore.load(path + '/LDA.model')
    dictionary = corpora.Dictionary.load(path + '/dictionary.dictionary')  # 加载字典
    corpus = corpora.MmCorpus(path + '/corpus.mm')  # 加载corpus
    topicMatrix = lda_multi[corpus]
    return topicMatrix

def constructPartionSkeleton(theEntList,theOrpanList,k):
    skeletonEntityList = [fileEnt for fileEnt in theEntList if
                          False == fileEnt.notUse and fileEnt.m_Id not in theOrpanList]
    for i in range(0, len(skeletonEntityList)): # normailze features
        skeletonEntityList[i].LdaVector = normalLdaTurples(topicMatrix[i], len(topicMatrix[i]))
    skeletonEntityList = [fileEnt for fileEnt in entityList if
                          False == fileEnt.notUse and fileEnt.m_Id not in OrpanEntityList]
    for i in range(0, len(skeletonEntityList)):
        skeletonEntityList[i].LdaVector = normalLdaTurples(topicMatrix[i], len(topicMatrix[i]))

    # outliers detector
    outDPre = OutLiersDetector(theEntList)
    theOrpanList.extend(outDPre.selectOrpanNodesPre(theOrpanList))  # applying LOF

    # filter - update - skeletonlist
    skeletonEntityList = [fileEnt for fileEnt in entityList if
                          False == fileEnt.notUse and fileEnt.m_Id not in OrpanEntityList]
    inputFeatures = [ent.LdaVector for ent in skeletonEntityList]
    labels = H_AgglomerativeClustering(inputFeatures, k)

    for i in range(len(skeletonEntityList)):
        skeletonEntityList[i].clusterId = labels[i]

    skeletonPartition = dict()
    for entity in skeletonEntityList:
        skeletonPartition.setdefault(entity.clusterId, []).append(entity.m_Id)
    skeletonPartitionWithsNums = [cluster for cluster in skeletonPartition.values()]
    return skeletonPartitionWithsNums

def mergePairs(finalClustersWithNums,theEntityList):
    finalClustersWithNames = convertNum2Names(finalClustersWithNums, fileNameList)
    for i in range(0, len(finalClustersWithNums)):
        for j in range(0, len(finalClustersWithNums[i])):
            curEntNum = finalClustersWithNums[i][j]
            if True == entityList[curEntNum].isMerge:
                finalClustersWithNums[i].append(theEntityList[curEntNum].m_mFileId)
                finalClustersWithNames[i].append(fileNameList[theEntityList[curEntNum].m_mFileId])
    return finalClustersWithNums,finalClustersWithNames

def propagateLabels():

    return
if __name__ == '__main__':

    OrpanEntityList = []               # init outlier entity
    entityList,FDG = initFileEntity()  # init file entity
    topicMatrix = getLDATopicMatrix(casePath)  # get sematic features (topic distribution matrix)
    topicNumber = len(topicMatrix[0])  # record topic number (different for dirreent cases)
    OutLiersDetector.__TOPIC__ = topicNumber

    # step 1 :  filter outliers and construct a partition skeleton
    for clusternum in range(2,40):  # select a best k according to the results
        skeletonPartitionWithsNums = constructPartionSkeleton(entityList,OrpanEntityList,clusternum)
        finalClustersWithNums = list()
        finalClustersWithNames = list()
        for cluster in skeletonPartitionWithsNums:
            finalClustersWithNums.append([entNum for entNum in cluster if entNum not in OrpanEntityList])
        # merge pairs of files together
        finalClustersWithNums,finalClustersWithNames =  mergePairs(finalClustersWithNums,entityList)


        # step 2 . label propagation to do
        # slpa rules can change accouding to the subject sysytems (user - assist)
        for i in FDG.nodes:
            FDG.nodes[i]['clusterNum'] = -1

        evWorker = Evaluator(topicNumber, FDG)
        Evaluator.__rsfPath__ = case + '.rsf'
        mojoResult = evWorker.getMojoResult(finalClustersWithNames, len(finalClustersWithNames))
        oaWorker = OutliersAssignment(FDG, 1000)
        finalClustersWithNums,lextremeNodes = oaWorker.slpa(finalClustersWithNums,OrpanEntityList,0,100)
        finalClustersWithNames = convertNum2Names(finalClustersWithNums, fileNameList)
        # # report the result
        evWorker = Evaluator(topicNumber, FDG)
        Evaluator.__rsfPath__=case+'.rsf'
        mojoResult = evWorker.getMojoResult(finalClustersWithNames, len(finalClustersWithNames))

