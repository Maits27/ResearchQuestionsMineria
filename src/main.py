import json

import pandas as pd
from loadSaveData import loadRAW, loadRAWwithClass, loadClassTextList, loadTextRaw
#from tokenization import tokenize, tokenizarSinLimpiar
from evaluation import classToClass, multiClassToClass, classDistribution, graficaMetrics, \
    graficoDeTodasLasEmocionesCM, calcular_metricas
import vectorization
from sentimentStats import print_number_distribution, plot_with_class_distribution
from reduceDataset import reduceDataset, takeThresDataset, reduceDataset2, crearMiniTests
import sys
import sentimentAnalysis





if __name__ == '__main__':
    if len(sys.argv) < 3:
        nInstances = 10000
        metricas=None
        extrasNombre = ''
        nTest = 1
        vectorsDimension = 768
        vectorizationMode = vectorization.bertTransformer
    else:
        nInstances = int(sys.argv[1])
        nTest = int(nInstances / 10)
        vectorsDimension = int(sys.argv[2])
        vectorizationMode = vectorization.bertTransformer

    ###################################################################
    ######################## REDUCCION DATASET #######################
    ###################################################################
    path = f'..\Datasets\Suicide_Detection_{nInstances}.csv'
    testPath = f'..\Datasets\Suicide_Detection_Test_5000.csv'

    # reduceDataset2(path, nInstances)
    # takeThresDataset(path, fiabilidad)
    # classDistribution(path, nInstances, extrasNombre)
    # classDistribution(testPath, 5000, 'test')
    # reduceDataset(path, nInstances, nTest)


    ###################################################################
    ######################## SENTIMENT ANALYSIS #######################
    ###################################################################

    # sentimentAnalysis.getSentiment(loadTextRaw(path), extrasNombre)
    # sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(path), 10000, extrasNombre)
    #
    # sentimentAnalysis.getSentiment(loadTextRaw(testPath), 'test')
    # sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(testPath), 5000, 'test')

    ####################### YA NO SE SI SIRVEN #######################
    # sentimientos = sentimentAnalysis.getArgmaxSentiment(nInstances)
    #sentimentAnalysis.getSentimentAndClassRoberta(loadRAWwithClass(path))
    ###################################################################

    # Matriz para mapear las clases con su emocion predominante:

    # classToClass(loadRAW(path), sentimientos, nInstances)

    # Gráficas de cada sentimiento con su distribución de clases correspondiente:
    # pathGrafica = f'../out/emociones/emocionesDominantesConClase_{nInstances}_{extrasNombre}.json'
    # distribution = print_number_distribution(pathGrafica)
    # plot_with_class_distribution(distribution, nInstances, extrasNombre)
    #
    # pathGrafica = f'../out/emociones/emocionesDominantesConClase_{5000}_test.json'
    # distribution = print_number_distribution(pathGrafica)
    # plot_with_class_distribution(distribution, 5000, 'test')

    ######################################################################
    ########################### TESTEO ###################################
    ######################################################################


    numTests = 5
    testsSize = len(loadRAW(path))/numTests
    # crearMiniTests(path, numTests)

    predicciones = {}
    resultado = {}

    data = loadRAWwithClass(testPath)
    sentimientos = sentimentAnalysis.getSentimentForTest(loadRAWwithClass(testPath))
    with open(f'../Predicciones/Emociones/Predicciones5000.json', 'r', encoding='utf-8') as json_file:
        inferencia = json.load(json_file)
    predicciones.update(inferencia)
    for id, texto in enumerate(sentimientos):
        prediccion = predicciones[str(id)]
        resultado[id] = texto[id]
        resultado[id].update({'prediccion': prediccion})
    with open(f'../Predicciones/Emociones/Predicciones_Con_ClaseySentimiento_Test5000.json', 'w', encoding='utf-8') as json_file:
        json.dump(resultado, json_file, indent=2, ensure_ascii=False)



    ###################################################################
    ################### MAPEO SENTIMIENTO-PREDICCION ##################
    ###################################################################
    predicciones = []
    clasesReales = []
    sentimientos = []
    path = f'../Predicciones/Emociones/Predicciones_Con_ClaseySentimiento_Test5000.json'
    with open(path, 'r', encoding='utf-8') as json_file:
        inferencia = json.load(json_file)
    for texto in inferencia:
        valores = inferencia[texto]
        predicciones.append(valores['prediccion'])
        clasesReales.append(valores['claseReal'])
        sentimientos.append(valores['emocion'])
    graficoDeTodasLasEmocionesCM(clasesReales, predicciones, sentimientos, 'Emociones')
    metricas = calcular_metricas(clasesReales, predicciones, sentimientos, 'Emociones')



    if metricas is None:
        with open(f'..\Predicciones\metricas_{extrasNombre}.json', 'r', encoding='utf-8') as json_file:
            metricas = json.load(json_file)

    nInst = []
    f1 = []
    acc = []
    fpr = []
    fnr = []

    for s in ['sadness', 'joy', 'anger', 'fear', 'love', 'surprise']:
        nInst.append(metricas[s]['nInstancias'])
        f1.append(metricas[s]['f1'])
        acc.append(metricas[s]['accuracy'])
        fpr.append(metricas[s]['false_positive_rate'])
        fnr.append(metricas[s]['false_negative_rate'])

    graficaMetrics(nInst, f1, fpr, fnr, extras='Emociones')

    # calcularMetricas(claseReal, clasePredicha)