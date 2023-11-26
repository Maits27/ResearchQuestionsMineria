import json

from src.loadSaveData import loadRAW
import numpy as np
import matplotlib.pyplot as plt



def plot_with_class_distribution(distribution, nInst, balanced=''):
    groups = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    group_count = {
        'suicide': [distribution[group]['suicide'] for group in groups],
        'non-suicide': [distribution[group]['non-suicide'] for group in groups]
    }

    width = 0.8


    colors = ['#3589D3', '#35DA79']  # Define colors for each class

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(groups))

    for clase, counts, color in zip(group_count.keys(), group_count.values(), colors):
        ax.bar(groups, counts, width, label=clase, bottom=bottom, color=color)
        bottom += counts


    ax.legend(loc="upper right", fontsize=15)

    plt.xticks(groups)
    plt.xlabel('Sentence main feeling')
    plt.ylabel('Number of texts')
    plt.savefig(f'..\img\FeelingDistribution\ClassDistribution_{nInst}_{balanced}.png')
    plt.show()

def print_number_distribution(path):
    distribution = {}
    for emocion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
        distribution[emocion] = {'suicide': 0, 'non-suicide': 0}
    with open(path, 'r', encoding='utf-8') as json_file:
        emociones_array = json.load(json_file)
    for texto in emociones_array:
        if texto['clase'].__eq__('suicide'):
            distribution[texto['emocion']]['suicide'] += 1
        else:
            distribution[texto['emocion']]['non-suicide'] += 1
    return distribution


if __name__ == '__main__':
    # data = loadRAW('../Datasets/Suicide_Detection_2000_Balanceado.csv')

    # instances_lengths = []
    # for text in data['text']:
    #     instances_lengths.append(len(text.split()))

    # print('Max words instance:', max(instances_lengths))
    # print('Min words instance:', min(instances_lengths))
    # print('Average words:', np.mean(instances_lengths))
    # print('Standard deviation:', np.std(instances_lengths))
    # path = '../out/emo/emocionesDominantesConClase_10000.json'
    # distribution = print_number_distribution(path)
    # plot_with_class_distribution(distribution)
    pass