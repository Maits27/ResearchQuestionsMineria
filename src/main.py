from loadSaveData import loadRAW, loadRAWwithClass, loadClassTextList, loadTextRaw
from tokenization import tokenize, tokenizarSinLimpiar
from evaluation import classToClass, multiClassToClass, classDistribution
import vectorization
from reduceDataset import reduceDataset, takeThresDataset2
import sys
import sentimentAnalysis


def preProcess(nInstances, vectorsDimension, vectorizationMode):
    path = f'Datasets\Suicide_Detection{nInstances}.csv' # Previosuly reduced with reduceDataset.py
    rawData = loadRAW(path)
    rawDataWithClass = loadRAWwithClass(path)
    if vectorizationMode != vectorization.bertTransformer:
        textosToken = tokenize(rawData)
        textEmbeddings = vectorizationMode(textosToken=textosToken, dimensiones=vectorsDimension)
    else: 
        textEmbeddings = vectorizationMode(rawData)
    return rawData, rawDataWithClass, textEmbeddings



def evaluate(rawData, rawDataWithClass, clusters, numClusters):
    tokensSinLimpiar = tokenizarSinLimpiar(rawDataWithClass) 

    # evaluation.classToCluster(rawDataWithClass, clusters)
    # evaluation.wordCloud(clusters, tokensSinLimpiar)
    # evaluation.getClusterSample(clusterList=clusters,
    #                             numClusters=numClusters,
    #                             rawData=rawData,
    #                             sample=5)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        nInstances = 20000
        nTest = 1
        vectorsDimension = 768
        vectorizationMode = vectorization.bertTransformer
    else:
        nInstances = int(sys.argv[1])
        nTest = int(nInstances / 10)
        vectorsDimension = int(sys.argv[2])
        vectorizationMode = vectorization.bertTransformer

    fiabilidad = 0.9

    path = f'..\Datasets\Suicide_Detection_thres_{fiabilidad}_parte2.csv'

    classDistribution(path, fiabilidad)

    reduceDataset(path, nInstances, nTest)

    path = f'..\Datasets\Suicide_Detection_train{nInstances}(test{nTest}).csv'

    sentimentAnalysis.getSentiment(loadTextRaw(path))
    sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(path), nInstances)
    sentimientos = sentimentAnalysis.getArgmaxSentiment(nInstances)

    classToClass(loadRAW(path), sentimientos, nInstances)
    #clusters, numClusters = executeClustering(clusteringAlgorithm, epsilon, minPts)
    #evaluate(rawData, rawDataWithClass, clusters, numClusters)