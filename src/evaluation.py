import matplotlib.pyplot as plt
import numpy as np
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

def classToClass(dataClass, sentimientos):
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
    plt.savefig(f'img\InitialMatrix')
    plt.show()

