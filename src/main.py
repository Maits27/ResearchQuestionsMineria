import json

import pandas as pd

from loadSaveData import loadRAW, loadRAWwithClass, loadClassTextList, loadTextRaw
from tokenization import tokenize, tokenizarSinLimpiar
from evaluation import classToClass, multiClassToClass, classDistribution, classToClassPorEmocion
import vectorization
from reduceDataset import reduceDataset, takeThresDataset, reduceDataset2, crearMiniTests
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
        nInstances = 2000
        nTest = 1
        vectorsDimension = 768
        vectorizationMode = vectorization.bertTransformer
    else:
        nInstances = int(sys.argv[1])
        nTest = int(nInstances / 10)
        vectorsDimension = int(sys.argv[2])
        vectorizationMode = vectorization.bertTransformer

    fiabilidad = 0
    path = f'..\Datasets\Suicide_Detection.csv'

    #reduceDataset2(path, nInstances)

    #takeThresDataset(path, fiabilidad)

    # path = f'..\Datasets\Suicide_Detection_thres_{fiabilidad}_parte2.csv'
    #
    #classDistribution(path, fiabilidad)
    #
    # reduceDataset(path, nInstances, nTest)
    #
    # path = f'..\Datasets\Suicide_Detection_train{nInstances}(test{nTest}).csv'
    #
    path = f'..\Datasets\Suicide_Detection_2000_Balanceado2.csv'
    sentimentAnalysis.getSentiment(loadTextRaw(path))
    sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(path), nInstances)
    # sentimientos = sentimentAnalysis.getArgmaxSentiment(nInstances)
    #sentimentAnalysis.getSentimentAndClassRoberta(loadRAWwithClass(path))
    #
    # classToClass(loadRAW(path), sentimientos, nInstances)
    # #clusters, numClusters = executeClustering(clusteringAlgorithm, epsilon, minPts)
    # #evaluate(rawData, rawDataWithClass, clusters, numClusters)


    ######################################################################
    ########################### TESTEO ###################################
    ######################################################################

    pathPrincipal = f'..\Datasets\Suicide_Detection_2000_Balanceado.csv'

    numTests = 5
    testsSize = len(loadRAW(pathPrincipal))/numTests
    # crearMiniTests(path, numTests)

    predicciones = {}
    resultado = {}

    # for i in range(numTests):
    #     path = f'Suicide_Detection_test{i}_400.0.csv'
    #     data = loadRAWwithClass(path)
    #     sentimientos = sentimentAnalysis.getSentimentForTest(loadRAWwithClass(path))
    #     with open(f'../Predicciones/Predicciones{i}{int(testsSize)}.json', 'r', encoding='utf-8') as json_file:
    #         inferencia = json.load(json_file)
    #     predicciones.update(inferencia)
    #     for id, texto in enumerate(sentimientos):
    #         prediccion = predicciones[str(id)]
    #         resultado[id] = texto[id]
    #         resultado[id].update({'prediccion': prediccion})
    #     with open(f'../Predicciones/Predicciones_Test{i}_{testsSize}.json', 'w', encoding='utf-8') as json_file:
    #         json.dump(resultado, json_file, indent=2, ensure_ascii=False)

    # predicciones = []
    # clasesReales = []
    # sentimientos = []
    # for i in range(numTests):
    #     path = f'../Predicciones/Predicciones_Test{i}_{testsSize}.json'
    #     with open(path, 'r', encoding='utf-8') as json_file:
    #         inferencia = json.load(json_file)
    #     for texto in inferencia:
    #         valores = inferencia[texto]
    #         predicciones.append(valores['prediccion'])
    #         clasesReales.append(valores['claseReal'])
    #         sentimientos.append(valores['emocion'])
    # for emocion in ['surprise']:
    #     classToClassPorEmocion(clasesReales, predicciones, sentimientos, emocion)

