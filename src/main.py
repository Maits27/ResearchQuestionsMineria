from loadSaveData import loadRAW, loadRAWwithClass
from tokenization import tokenize, tokenizarSinLimpiar
import vectorization
import sys


def preProcess(nInstances, vectorsDimension, vectorizationMode):
    path = f'../Datasets/Suicide_Detection{nInstances}.csv' # Previosuly reduced with reduceDataset.py
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
        nInstances = 1000
        vectorsDimension = 768
        vectorizationMode = vectorization.bertTransformer
    else:
        nInstances = int(sys.argv[1])
        vectorsDimension = int(sys.argv[2])
        if sys.argv[3] == 'doc2vec':
            vectorizationMode = vectorization.doc2vec
        elif sys.argv[3] == 'bert':
            vectorizationMode = vectorization.bertTransformer
    epsilon = float(sys.argv[5])
    minPts = int(sys.argv[6])

    rawData, rawDataWithClass, textEmbeddings = preProcess(nInstances, vectorsDimension, vectorizationMode)
    clusters, numClusters = executeClustering(clusteringAlgorithm, epsilon, minPts)
    evaluate(rawData, rawDataWithClass, clusters, numClusters)