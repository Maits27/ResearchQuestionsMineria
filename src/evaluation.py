import matplotlib.pyplot as plt
import numpy as np
from loadSaveData import loadRAW
from sklearn.metrics import confusion_matrix, adjusted_rand_score, jaccard_score, fowlkes_mallows_score, silhouette_score
from wordcloud import WordCloud
from tqdm import tqdm

def clase_a_num(data):
    clases = data['class'].copy()
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
    plt.savefig(f'..\img\InitialMatrix_{nInstancias}')
    plt.show()




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
    plt.savefig(f'..\img\InitialMultiClassMatrix_ConFiabilidad_{fiabilidad}.png')
    plt.show()


def classDistribution(path, fiabilidad):
    df = loadRAW(path)

    # Count the number of instances of each class
    suicide_count = df['class'].value_counts()['suicide']
    non_suicide_count = df['class'].value_counts()['non-suicide']

    print(suicide_count+non_suicide_count)
    print(suicide_count)
    print(non_suicide_count)

    # Create a bar chart
    labels = ['Suicide', 'Non-Suicide']
    counts = [suicide_count, non_suicide_count]

    plt.bar(labels, counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.title(f"Class distribution Matrix Fiabilidad {fiabilidad}")
    plt.savefig(f'..\img\ClassDistribution_ConFiabilidad_{fiabilidad}_{suicide_count+non_suicide_count}_Instanccias.png')
    plt.show()