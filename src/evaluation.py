import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import seaborn as sns

from loadSaveData import loadRAW
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, \
    precision_recall_fscore_support
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

    plt.figure(figsize=(12, 6))

    # Create a bar chart
    labels = ['Suicide', 'Non-Suicide']
    counts = [suicide_count, non_suicide_count]
    colors = ['#3589D3', '#3589D3']

    plt.bar(labels, counts, color=colors)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
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
def calcular_metricas(clasesReales, predicciones, modelo):
    archivo_salida = f'../Predicciones/General/MetricasTodosModelos.json'
    if not Path(archivo_salida).is_file():
        metricas = {}
    else:
        with open(archivo_salida, 'r', encoding='utf-8') as json_file:
            metricas = json.load(json_file)
    p = clase_a_num2(predicciones)
    cr = clase_a_num2(clasesReales)
    matriz_confusion = confusion_matrix(p, cr)

    true_positive = matriz_confusion[1, 1]
    true_negative = matriz_confusion[0, 0]
    false_positive = matriz_confusion[0, 1]
    false_negative = matriz_confusion[1, 0]

    precision, recall, fscore, support = precision_recall_fscore_support(cr, p, average='weighted')

    nInst = len(clasesReales)

    # Calcular métricas
    accuracy = (true_positive + true_negative) / np.sum(matriz_confusion)
    false_positive_rate = false_positive / (false_positive + true_negative)
    false_negative_rate = false_negative / (false_negative + true_positive)

    metricas[modelo] = {'nombre': modelo, 'nInstancias': int(nInst), 'metricas': {'f1': fscore, 'precision': precision, 'recall': recall,
                        'accuracy': accuracy, 'false_positive_rate': false_positive_rate, 'false_negative_rate': false_negative_rate}}

    with open(archivo_salida, 'w', encoding='utf-8') as json_file:
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


def graficaDeMetricasPorModelo(path, cuales):
    with open(path, 'r', encoding='utf-8') as json_file:
        modelos = json.load(json_file)

    n_metricas = ['f1', 'recall', 'precision', 'false_positive_rate', 'false_negative_rate']
    nombres = ['Weighted F-Score', 'W-Recall', 'W-Precision', 'FPR', 'FNR']
    plt.figure(figsize=(15, 5))

    colores = ['red', 'orange', 'green', 'blue', 'purple', 'pink']

    for i, c in enumerate(cuales):
        metricas = []
        modelo = modelos[c]
        # nombres.append(modelo['nombre'])
        if c.__eq__('Base'): labelNombre = 'CardiffNLP (primeras 10000 instancias)'
        elif c.__eq__('Emociones'): labelNombre = 'Textos + Valor de las emociones'
        elif c.__eq__('RandomGuesser'): labelNombre = 'CardiffNLP sin fine-tune'
        elif c.__eq__('SoloEmociones'): labelNombre = 'Solo valor de las emociones'
        else: labelNombre = 'CardiffNLP (distribución en base a las emociones predominantes)'

        for n in n_metricas:
            metricas.append(modelo['metricas'][n])
        plt.plot(nombres, metricas, label=f'{labelNombre}', marker='o', color=colores[i])

    # Añadir etiquetas y leyenda
    plt.xlabel('Métricas (siendo suicidio TP)')
    plt.ylabel('Valor de la métrica')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'..\img\Metricas\Metricas_TodosLosModelos')
    plt.savefig(f'..\Predicciones\Metricas_TodosLosModelos')
    plt.show()



"""
#########################################################################################################
########################################## LOSS FUNCTION ################################################
#########################################################################################################
"""

def documentoLoss(path, extras=''):
    pathSalida = f'../Predicciones/{extras}/Loss_Modelo_{extras}.json'

    with open(path, 'r', encoding='utf-8') as json_file:
        train_state = json.load(json_file)

    loss = []
    log_history = train_state['log_history']
    for i in range(0, len(log_history), 2):
        step1 = log_history[i]['step']
        train_loss = log_history[i]['loss']

        # Verificar que haya una clave siguiente antes de acceder a ella
        if i + 1 < len(log_history):
            step2 = log_history[i+1]['step']
            eval_loss = round(log_history[i+1]['eval_loss'], 4)
            runtime = log_history[i+1]['eval_runtime']

            if step1 == step2:
                loss.append({'step': step2, 'train_loss': train_loss, 'eval_loss': eval_loss, 'runtime': runtime})

    with open(pathSalida, 'w', encoding='utf-8') as json_file:
        json.dump(loss, json_file, indent=2, ensure_ascii=False)
    return loss


def graficaLoss(d1, d2, d3, d4):

    # Extraer los valores de cada lista para la gráfica
    nombres = ['Base', 'Emociones', 'Solo Emociones', 'Largo']
    for i, d in enumerate([d1, d2, d3, d4]):
        if d is not None:
            steps = [item["step"] for item in d]
            eval_losses = [item["eval_loss"] for item in d]
            train_losses = [item["train_loss"] for item in d]
            plt.plot(steps, eval_losses, label=f'Eval Loss {nombres[i]}', marker='o')
            plt.plot(steps, train_losses, label=f'Train Loss {nombres[i]}', marker='o')
    # Agregar etiquetas y leyenda
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()

    # Mostrar la gráfica
    plt.show()

def graficaLossMultiple(d1, d2, d3, d4):
    nombres = ['Base', 'Emociones', 'Solo Emociones', 'Sin Clase']

    fig, axs = plt.subplots(2,2, figsize=(15, 6))

    kont = 0
    runs = []
    for i, d in enumerate([d1, d2, d3, d4]):
        if d is not None:
            steps = [item["step"] for item in d]
            eval_losses = [item["eval_loss"] for item in d]
            train_losses = [item["train_loss"] for item in d]
            runtime = [item["runtime"] for item in d]
            runs.append(runtime)
            x=0
            y=0
            if kont == 1 or kont == 3: x = 1
            if kont == 2 or kont == 3: y = 1

            axs[x, y].plot(steps, eval_losses, label=f'Eval Loss {nombres[i]}', marker='o', color='#3589D3')
            axs[x, y].plot(steps, train_losses, label=f'Train Loss {nombres[i]}', marker='o', color='#35DA79')
            axs[x, y].set_title(f'Loss del modelo {nombres[i]}')
            axs[x, y].set_xlabel('Step')
            axs[x, y].set_ylabel('Loss')
            axs[x, y].legend()

            kont += 1

    #Ajustar el diseño para evitar solapamiento
    plt.tight_layout()

    plt.savefig(f'..\img\Metricas\LossFunctions')
    plt.show()


if __name__ == '__main__':
    pass