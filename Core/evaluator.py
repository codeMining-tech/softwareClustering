import subprocess

import networkx as nx

#from core.MojoFM import exportingToMoJoFormatAlgorithm

def exportingToMoJoFormatAlgorithm(k, n, clustersFinal, fileNames, filePathNames, run_no, folderPathResult):

    f1 = open(folderPathResult + "/MoJoAlgorithm" + str(k) + "_" + str(run_no) + ".txt", "w")
    for clusNum in range(0,k):
        for i in range(len(clustersFinal[clusNum])):
            f1.write("contain ")
            f1.write("Clu")
            f1.write(str(clusNum))
            f1.write(" ")
            f1.write(clustersFinal[clusNum][i])
            f1.write("\n")
    f1.close()
class Evaluator():
    __rsfPath__ = '/intl.rsf'
    __rsfPath__ = '/extensions.rsf'
    __rsfPath__ = '/dom.rsf'
    def __init__(self,theTopicNum,FDG):
        self.m_tpoicNum = theTopicNum
        self.m_FDG = FDG
    def getMojoResult(self,clustersWithNames,k):

        rsfPath = Evaluator.__rsfPath__
        # rsfPath = '/intl.rsf'
        # rsfPath = '/itk-modules.rsf'

        pathResultString = []
        pathResultString.append("result/")
        run_no = self.m_tpoicNum
        exportingToMoJoFormatAlgorithm(k, 0, clustersWithNames, None, None, run_no, pathResultString[0])
        # calculateMoJo(k,0)

        MoJoAlgorithmString = pathResultString[0] + "/MoJoAlgorithm" + str(k) + "_" + str(
            run_no) + ".txt"
        proc = subprocess.Popen(
            ["java", "mojo/MoJo", MoJoAlgorithmString, pathResultString[0] + rsfPath],
            stdout=subprocess.PIPE)

        outs, errs = proc.communicate()

        mojoMeasure = int(outs[:-1])
        proc = subprocess.Popen(
            ["java", "mojo/MoJo", MoJoAlgorithmString, pathResultString[0] + rsfPath,
             "-fm"],
            stdout=subprocess.PIPE)
        outs, errs = proc.communicate()
        mojoFmMeasure = float(outs[:-1])
        print(
            '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>mojoMeasure :',
            mojoMeasure)
        print(
            '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>mojoFmMeasure :',
            mojoFmMeasure)
        return mojoMeasure, mojoFmMeasure

    def getExpertPartition(self,theSrcFileList):
        pathResultString = []
        pathResultString.append("result")
        rsfF = open(pathResultString[0]+self.__rsfPath__,'r')
        clusters = dict()
        for line in rsfF.readlines():
            line =line.strip('\n')
            info = line.split(' ')
            clusters.setdefault(info[1],[]).append(info[2])
        resultWithNums = []
        for cluster in clusters:
            resultWithNums.append([theSrcFileList.index(clusterName) for clusterName in clusters[cluster] if clusterName in theSrcFileList])
        return resultWithNums



