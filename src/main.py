import json

import numpy as np
import pandas as pd
from loadSaveData import loadRAW, loadRAWwithClass, loadClassTextList, loadTextRaw
#from tokenization import tokenize, tokenizarSinLimpiar
from evaluation import classToClass, multiClassToClass, classDistribution, graficaMetrics, \
    graficoDeTodasLasEmocionesCM, calcular_metricas, documentoLoss, graficaLoss, graficaLossMultiple, \
    graficaDeMetricasPorModelo

from sentimentStats import print_number_distribution, plot_with_class_distribution
from reduceDataset import reduceDataset, takeThresDataset, reduceDataset2, crearMiniTests
import sys
import sentimentAnalysis

def crearLoss(pathBase='', pathEmociones='', pathSoloEmociones='', pathSinClase=''):
    loss1, loss2, loss3, loss4 = None, None, None, None
    if not pathBase.__eq__(''):
        loss1 = documentoLoss(pathBase, 'Base')
    if not pathEmociones.__eq__(''):
        loss2 = documentoLoss(pathEmociones, 'Emociones')
    if not pathSoloEmociones.__eq__(''):
        loss3 = documentoLoss(pathSoloEmociones, 'SoloEmociones')
    if not pathSinClase.__eq__(''):
        loss4 = documentoLoss(pathSinClase, 'SinClase')
def loss(pathBase='', pathEmociones='', pathSoloEmociones='', pathSinClase=''):
    loss1, loss2, loss3, loss4 = None, None, None, None
    if not pathBase.__eq__(''):
        with open(pathBase, 'r', encoding='utf-8') as json_file:
            loss1 = json.load(json_file)
    if not pathEmociones.__eq__(''):
        with open(pathEmociones, 'r', encoding='utf-8') as json_file:
            loss2 = json.load(json_file)
    if not pathSoloEmociones.__eq__(''):
        with open(pathSoloEmociones, 'r', encoding='utf-8') as json_file:
            loss3 = json.load(json_file)
    if not pathSinClase.__eq__(''):
        with open(pathSinClase, 'r', encoding='utf-8') as json_file:
            loss4 = json.load(json_file)
    #
    #     with open(f'../Predicciones/Emociones/Loss_Modelo_Emociones.json', 'r', encoding='utf-8') as json_file:
    #         loss2 = json.load(json_file)
    if loss1 is not None or loss2 is not None or loss3 is not None or loss4 is not None:
        # graficaLoss(loss1, loss2, loss3, loss4)
        graficaLossMultiple(loss1, loss2, loss3, loss4)

def crearJSONConPredClase(testPath, model):

    predicciones = {}
    resultado = {}
    sentimientos = sentimentAnalysis.getSentimentForTest(loadRAWwithClass(testPath))
    with open(f'../Predicciones/{model}/Predicciones5000.json', 'r', encoding='utf-8') as json_file:
        inferencia = json.load(json_file)
    predicciones.update(inferencia)
    for id, texto in enumerate(sentimientos):
        prediccion = predicciones[str(id)]
        resultado[id] = texto[id]
        resultado[id].update({'prediccion': prediccion})
    with open(f'../Predicciones/{model}/Predicciones_Con_ClaseySentimiento_Test5000.json', 'w', encoding='utf-8') as json_file:
        json.dump(resultado, json_file, indent=2, ensure_ascii=False)


def sacarMetricas(testPath, modelos):
    for modelo in modelos:  # TODO SOLO EMOCIONES Y LARGO
        predicciones = []
        clasesReales = []
        path = f'../Predicciones/{modelo}/Predicciones_Con_ClaseySentimiento_Test5000.json'
        with open(path, 'r', encoding='utf-8') as json_file:
            inferencia = json.load(json_file)
        for texto in inferencia:
            valores = inferencia[texto]
            predicciones.append(valores['prediccion'])
            clasesReales.append(valores['claseReal'])
            np.save('ClaseRealTest5000.npy', np.array(clasesReales))

        calcular_metricas(clasesReales, predicciones, modelo)
    graficaDeMetricasPorModelo('../Predicciones/General/MetricasTodosModelos.json', modelos)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        nInstances = 10000
        metricas=None
        extrasNombre = ''
        nTest = 1
        vectorsDimension = 768
    else:
        nInstances = int(sys.argv[1])
        nTest = int(nInstances / 10)
        vectorsDimension = int(sys.argv[2])

    path = f'..\Datasets\Suicide_Detection_{nInstances}.csv'
    path2 = f'..\Datasets\Suicide_Detection_10000_Balanceado_SinContarClases.csv'
    testPath = f'..\Datasets\Suicide_Detection_Test_5000.csv'

    ###################################################################
    ######################## REDUCCION DATASET #######################
    ###################################################################

    # reduceDataset2(path, nInstances)
    # takeThresDataset(path, fiabilidad)
    # classDistribution(path, nInstances, extrasNombre)
    # classDistribution(testPath, 5000, 'test')
    # reduceDataset(path, nInstances, nTest)

    ###################################################################
    ######################## SENTIMENT ANALYSIS #######################
    ###################################################################
    # path = f'..\Datasets\Suicide_Detection_10000_Balanceado_SinContarClases.csv'
    # sentimentAnalysis.getSentiment(loadTextRaw(path), 'Balanceado')
    # sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(path), 10000, 'Balanceado')
    #
    # sentimentAnalysis.getSentiment(loadTextRaw(testPath), 'test')
    # sentimentAnalysis.getArgmaxSentimentAndClass(loadRAWwithClass(testPath), 5000, 'test')

    # # Matriz para mapear las clases con su emocion predominante:
    # sentimientos = sentimentAnalysis.getArgmaxSentiment(nInstances)
    # sentimentAnalysis.getSentimentAndClassRoberta(loadRAWwithClass(path))
    # classToClass(loadRAW(path), sentimientos, nInstances)

    # Gráficas de cada sentimiento con su distribución de clases correspondiente:

    # pathGrafica = f'../out/emociones/emocionesDominantesConClase_{nInstances}_.json'
    # distribution = print_number_distribution(pathGrafica)
    # plot_with_class_distribution(distribution, nInstances, extrasNombre)
    #
    # pathGrafica = f'../out/emociones/emocionesDominantesConClase_{nInstances}_Balanceado.json'
    # distribution = print_number_distribution(pathGrafica)
    # plot_with_class_distribution(distribution, nInstances, 'Balanceado')
    #
    # pathGrafica = f'../out/emociones/emocionesDominantesConClase_{5000}_test.json'
    # distribution = print_number_distribution(pathGrafica)
    # plot_with_class_distribution(distribution, 5000, 'test')

    ######################################################################
    ########################### TESTEO ###################################
    ######################################################################

    for modelo in ['SoloEmociones']:
        crearJSONConPredClase(testPath, modelo)

    crearLoss(pathBase='../Predicciones/Base/trainer_state.json',
              pathEmociones='../Predicciones/Emociones/trainer_state.json',
              pathSoloEmociones='../Predicciones/SoloEmociones/trainer_state.json',
              pathSinClase='../Predicciones/SinClase/trainer_state.json')
    loss(pathBase='../Predicciones/Base/Loss_Modelo_Base.json',
         pathEmociones='../Predicciones/Emociones/Loss_Modelo_Emociones.json',
         pathSoloEmociones='../Predicciones/SoloEmociones/Loss_Modelo_SoloEmociones.json',
         pathSinClase='../Predicciones/SinClase/Loss_Modelo_SinClase.json')

    sacarMetricas(testPath, ['RandomGuesser', 'Base', 'Emociones', 'SoloEmociones', 'SinClase'])
