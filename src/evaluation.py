import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from loadSaveData import loadRAW
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt

"""
########################################################################################################################
############################################# PASAR CLASES A NUMEROS ###################################################
########################################################################################################################
"""
def clase_a_num(data):
    clases = data['class'].copy()
    res = []
    for i, c in enumerate(clases):
        if c.__eq__('suicide'):
            res.append(0)
        else:
            res.append(1)
    return res
def clase_a_num2(clases):
    res = []
    for i, c in enumerate(clases):
        if c.__eq__('suicide'):
            res.append(0)
        else:
            res.append(1)
    return res


def sentimiento_a_num(data):
    res = []
    for i, c in enumerate(data):
        if c.__eq__('sadness'):
            res.append(0)
        elif c.__eq__('joy'):
            res.append(1)
        elif c.__eq__('love'):
            res.append(2)
        elif c.__eq__('anger'):
            res.append(3)
        elif c.__eq__('fear'):
            res.append(4)
        else:
            res.append(5)
    return res
def todo_a_num(data, sent):
    c = data['class'].copy()
    sentimientos = []
    clases = []
    for i, array_s in enumerate(sent):
        if c[i].__eq__('suicide'):
            clase_texto = 0
        else:
            clase_texto = 1
        for s in array_s:
            clases.append(clase_texto)
            if s.__eq__('sadness'):
                sentimientos.append(0)
            elif s.__eq__('joy'):
                sentimientos.append(1)
            elif s.__eq__('love'):
                sentimientos.append(2)
            elif s.__eq__('anger'):
                sentimientos.append(3)
            elif s.__eq__('fear'):
                sentimientos.append(4)
            else:
                sentimientos.append(5)
    return sentimientos, clases

"""
########################################################################################################################
############################################# TIPOS DE EVALUACION ######################################################
########################################################################################################################
"""

"""
#########################################################################################################
######################################## CLASS TO FEELING MATRIX ########################################
#########################################################################################################
"""
def classToClass(dataClass, sentimientos, nInstancias):
    """
    data: tiene que ser un dataframe con un campo 'text' y 'class'
    """
    cm = confusion_matrix(sentimiento_a_num(sentimientos), clase_a_num(dataClass))
    # Supongamos que tienes 20 grupos y 2 clases
    num_groups = len(set(sentimientos))
    num_classes = len(nombres_clases := ['suicide', 'non-suicide'])
    nombres_sentimientos = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    if num_classes < num_groups:
        cm = cm[:, :num_classes]
    elif num_groups < num_classes:
        cm = cm[:num_classes, :]

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, aspect='auto', interpolation='nearest', vmin=0, vmax=2000)

    # Personalizar el eje x y el eje y para mostrar los grupos y las clases
    plt.xticks(np.arange(num_classes), [f'Class {nombres_clases[i]}' for i in range(num_classes)])
    plt.yticks(np.arange(num_groups), [f' {sentimiento}' for sentimiento in nombres_sentimientos])
    thresh = cm.max() / 2.

    for i in range(len(nombres_sentimientos)):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i][j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    # Etiquetas para los ejes
    plt.xlabel("Class")
    plt.ylabel("Emotion")

    plt.title("Class2Class Matrix")
    plt.savefig(f'..\img\FeelingMatrix\InitialMatrix_{nInstancias}')
    plt.show()



def multiClassToClass(dataClass, sentimientos, fiabilidad):
    """
    data: tiene que ser un dataframe con un campo 'text' y 'class'
    """
    num_sen, num_class = todo_a_num(dataClass, sentimientos)
    cm = confusion_matrix(num_sen, num_class)
    # Supongamos que tienes 20 grupos y 2 clases
    nombres_sentimientos = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    num_groups = len(nombres_sentimientos)
    num_classes = len(nombres_clases := ['suicide', 'non-suicide'])


    if num_classes < num_groups:
        cm = cm[:, :num_classes]
    elif num_groups < num_classes:
        cm = cm[:num_classes, :]

    plt.figure(figsize=(10, 6))
    plt.imshow(cm, cmap=plt.cm.Blues, aspect='auto', interpolation='nearest', vmin=0, vmax=2000)

    # Personalizar el eje x y el eje y para mostrar los grupos y las clases
    plt.xticks(np.arange(num_classes), [f'Class {nombres_clases[i]}' for i in range(num_classes)])
    plt.yticks(np.arange(num_groups), [f' {sentimiento}' for sentimiento in nombres_sentimientos])
    thresh = cm.max() / 2.

    for i in range(len(nombres_sentimientos)):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i][j], 'd'), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    # Etiquetas para los ejes
    plt.xlabel("Class")
    plt.ylabel("Emotion")

    plt.title(f"Class2Class Matrix Fiabilidad {fiabilidad}")
    plt.savefig(f'..\img\FeelingMatrix\InitialMultiClassMatrix_ConFiabilidad_{fiabilidad}.png')
    plt.show()

"""
#########################################################################################################
############################ DISTRIBUCION CLASE SUICIDIO NO SUICIDIO ####################################
#########################################################################################################
"""

def classDistribution(path, nInst, extrasNombre=''):
    df = loadRAW(path)

    # Count the number of instances of each class
    suicide_count = df['class'].value_counts()['suicide']
    non_suicide_count = df['class'].value_counts()['non-suicide']

    print(nInst)
    print(suicide_count)
    print(non_suicide_count)

    # Create a bar chart
    labels = ['Suicide', 'Non-Suicide']
    counts = [suicide_count, non_suicide_count]

    plt.bar(labels, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title(f"Class distribution Matrix {nInst} instances")
    plt.savefig(f'..\img\ClassDistribution\ClassDistribution_{nInst}_{extrasNombre}.png')
    plt.show()








"""
#########################################################################################################
############################ MATRICES DE CONFUSION POR SENTIMIENTO ######################################
#########################################################################################################
"""

def conseguirMatrizEmocion(clasesReales, predicciones, sentimientos, emocion):
    c = []
    p = []
    for i, sentimiento in enumerate(sentimientos):
        if sentimiento.__eq__(emocion):
            c.append(clasesReales[i])
            p.append(predicciones[i])

    cm = confusion_matrix(clase_a_num2(p), clase_a_num2(c))
    return cm



def graficoDeTodasLasEmocionesCM(clasesReales, predicciones, sentimientos, extrasNombre=''):

    matrices_confusion = []
    for emocion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
        cm = conseguirMatrizEmocion(clasesReales, predicciones, sentimientos, emocion)
        matrices_confusion.append(cm)

    # Crea una figura y subgráficos (2 filas, 3 columnas)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Itera sobre las matrices de confusión y los ejes correspondientes
    for i, matriz_confusion in enumerate(matrices_confusion):
        fila = i // 3
        columna = i % 3
        ax = axs[fila, columna]

        # Visualiza la matriz de confusión
        im = ax.imshow(matriz_confusion, interpolation='nearest', cmap=plt.cm.Blues)

        # Añade etiquetas, título, etc.
        ax.set(xticks=np.arange(matriz_confusion.shape[1]),
               yticks=np.arange(matriz_confusion.shape[0]),
               xticklabels=["Clase 0", "Clase 1"],
               yticklabels=["Clase 0", "Clase 1"],
               title=f'Matriz de Confusión {i + 1}',
               ylabel='Etiqueta Verdadera',
               xlabel='Predicción')

        # Añade los valores en cada celda
        for x in range(matriz_confusion.shape[0]):
            for y in range(matriz_confusion.shape[1]):
                thresh = matriz_confusion.max() / 2.
                ax.text(y, x, str(matriz_confusion[x, y]), ha="center", va="center", color="white" if matriz_confusion[x, y] > thresh else "black")

    # Ajusta el diseño de los subgráficos
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Añade una barra de color
    cbar = fig.colorbar(im, ax=axs, orientation='vertical')

    # Añade un título general
    plt.suptitle('Matrices de Confusión', fontsize=16)
    plt.savefig(f'..\img\PredictionC2C\TODASLASMATRICES_{len(predicciones)}_{extrasNombre}')
    # Muestra el gráfico
    plt.show()



"""
#########################################################################################################
############################################# METRICAS ##################################################
#########################################################################################################
"""
# def calcularMetricas(claseReal, clasePredicha):
#     # Metrics
#     accuracy = accuracy_score(claseReal, clasePredicha)
#     confusion = confusion_matrix(claseReal, clasePredicha)
#     report = classification_report(claseReal, clasePredicha)
#     kappa = cohen_kappa_score(claseReal, clasePredicha)
#
#     print(f'Accuracy: {accuracy}')
#     print(f'Confusion matrix\n{confusion}')
#     print(report)
#     print(f'Kappa: {kappa}')


def calcular_metricas(clasesReales, predicciones, sentimientos, extrasNombre=''):
    metricas = {}
    for emocion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
        matriz_confusion = conseguirMatrizEmocion(clasesReales, predicciones, sentimientos, emocion)

        true_positive = matriz_confusion[1, 1]
        true_negative = matriz_confusion[0, 0]
        false_positive = matriz_confusion[0, 1]
        false_negative = matriz_confusion[1, 0]

        nInst=true_positive+true_negative+false_negative+false_positive

        # Calcular métricas
        accuracy = (true_positive + true_negative) / np.sum(matriz_confusion)
        false_positive_rate = false_positive / (false_positive + true_negative)
        false_negative_rate = false_negative / (false_negative + true_positive)

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        if precision + recall == 0:
            f1 = 0  # Evitar división por cero
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        metricas[emocion] = {'nInstancias': int(nInst), 'f1': f1, 'accuracy': accuracy, 'false_positive_rate': false_positive_rate,
                             'false_negative_rate': false_negative_rate}

        with open(f'..\Predicciones\metricas_{extrasNombre}.json', 'w', encoding='utf-8') as json_file:
            json.dump(metricas, json_file, indent=2, ensure_ascii=False)

    return metricas
def graficaMetrics(instanciasPorEmocion, fscore, false_positive_rate, false_negative_rate, extras=''):
    # Datos de ejemplo
    sentimientos = ['sadness', 'joy', 'anger', 'fear', 'love', 'surprise']

    # Crear la figura y los ejes
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # Crear la gráfica de barras para el número de instancias de entrenamiento por idioma
    bars = ax1.bar(sentimientos, instanciasPorEmocion, color='#3589D3', label='Instancias de entrenamiento')
    ax1.set_xlabel('Sentimientos\n')

    ax1.set_ylabel('Número de instancias', color='black')
    ax1.tick_params(axis='y', labelcolor='#3589D3')


    # Rotar los nombres de los idiomas en el eje x
    plt.xticks(rotation=90)  # Rotar 90 grados

    for bar, instancia in zip(bars, instanciasPorEmocion):
        height = bar.get_height()
        ax1.annotate(f'{instancia}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # Ajustar la posición vertical del texto
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Crear la gráfica de línea para el número de instancias correctamente clasificadas (accuracy)
    ax2 = ax1.twinx()  # Crear un segundo eje y
    ax2.plot(sentimientos, fscore, color='#4AF50E', marker='o', label='Fscore')
    ax2.plot(sentimientos, false_positive_rate, color='#29067F', marker='x',
             label='False positive rate')  # Nueva línea de accuracy
    ax2.plot(sentimientos, false_negative_rate, color='#F52E0E', marker='s',
             label='False negative rate')  # Otra nueva línea de accuracy
    ax2.set_ylabel('Valor de la métrica', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Añadir leyendas
    plt.legend(loc='upper right', fontsize='small')

    # Mostrar el gráfico
    plt.title('Diferentes métricas en base a las emociones\n\n\n')
    plt.tight_layout()  # Ajustar diseño para evitar superposición
    plt.savefig(f'..\img\Metricas\Metricas_{sum(instanciasPorEmocion)}_{extras}')
    plt.show()


if __name__ == '__main__':
    pass