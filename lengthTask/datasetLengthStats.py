import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def plot_real_distribution(instances_lengths):
    plt.hist(instances_lengths, bins=range(0, 9685, 1))
    plt.title('Distribution on tweets word length')
    plt.xlabel('Tweet word length')
    plt.ylabel('Number of tweets')
    #plt.show()
    plt.savefig('datasetLengthStats/realDistribution.png')  


def plot_less_600_words_distribution(instances_lengths):
    less_600_instances_length = []  
    for length in instances_lengths:
        if length <= 600: less_600_instances_length.append(length)

    plt.hist(less_600_instances_length, bins=range(0, 601, 1))
    plt.title('Distribution on <600 word length tweet')
    plt.xlabel('Tweet word length')
    plt.ylabel('Number of tweets')
    #plt.show()
    plt.savefig('datasetLengthStats/realDistribution<600.png')  


def print_number_distribution(data):
    distribution = {}
    for index in data.index:
        length = len(data['text'][index])
        resto = length % 100
        cuantoHasta100 = 100 - resto
        divisorDe100Siguiente = length + cuantoHasta100
        __incrementValue(distribution, divisorDe100Siguiente, data['class'][index])


    for key in range(100, 2000, 100):
        print(f'<={key}: suicide: {distribution[key]["suicide"]}; non-suicide: {distribution[key]["non-suicide"]}')
    return distribution


def number_distribution_in_Xs(data, X):
    distribution = {}
    for index in data.index:
        length = len(data['text'][index])
        resto = length % X
        cuantoHasta100 = X - resto
        divisorDe100Siguiente = length + cuantoHasta100
        __incrementValue(distribution, divisorDe100Siguiente, data['class'][index])

    return distribution


def __incrementValue(dict, length, clase):
    if length not in dict: dict[length] = {'suicide': 0, 'non-suicide': 0}
    dict[length][clase] += 1


def box_plot_less_400(instances_lengths):
    less_400_instances_length = []  
    for length in instances_lengths:
        if length <= 400: less_400_instances_length.append(length)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.boxplot(less_400_instances_length, vert=False)
    plt.title('Boxplot on <400 word tweet length instances')
    plt.xlabel('Tweets word length')
    #plt.show()
    plt.savefig('datasetLengthStats/boxPlot<400.png')  


def plot_with_class_distribution_less_2000_words(distribution):
    groups = list(range(100, 2000, 100))
    group_count = {
        'suicide': [distribution[group]['suicide'] for group in groups],
        'non-suicide': [distribution[group]['non-suicide'] for group in groups]
    }

    width = 75

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(groups))

    for clase, counts in group_count.items():
        ax.bar(groups, counts, width, label=clase, bottom=bottom)
        bottom += counts

    ax.set_title(f"Class distribution on different tweet word lengths")
    ax.legend(loc="upper right")
    
    plt.xticks(list(range(100, 2000, 100)))
    plt.xlabel('Tweet word length')
    plt.ylabel('Number of tweets')
    #plt.show()
    plt.savefig('datasetLengthStats/classDistribution<2000.png')  



def plot_with_class_distribution_less_700_words(distribution):
    groups = list(range(100, 750, 50))
    group_count = {
        'suicide': [distribution[group]['suicide'] for group in groups],
        'non-suicide': [distribution[group]['non-suicide'] for group in groups]
    }

    width = 40

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(groups))

    for clase, counts in group_count.items():
        p = ax.bar(groups, counts, width, label=clase, bottom=bottom)
        bottom += counts

        class_percentajes = __get_percentaje_of_the_class_X_by_range(clase, distribution)
        ax.bar_label(p, labels=class_percentajes, label_type='center')

    ax.set_title(f"Class distribution on <=700 word length tweets")
    ax.legend(loc="upper right")
    
    plt.xticks(list(range(100, 750, 50)))
    plt.xlabel('Tweet word length')
    plt.ylabel('Number of tweets')
    #plt.show()
    plt.savefig('datasetLengthStats/classDistribution<700.png')  


def __get_percentaje_of_the_class_X_by_range(clase, distribution):
    # return [f'{round((value[clase] / (value["suicide"] + value["non-suicide"]))*100, 2)}%' for key, value in distribition.items() if key < 700]
    return_list = []
    keys = list(distribution.keys())
    keys.sort()
    for key in keys:
        if key < 700:
            total_de_instancias_en_grupo = distribution[key]["suicide"] + distribution[key]["non-suicide"]
            percentaje = round((distribution[key][clase] / total_de_instancias_en_grupo)*100, 2)
            return_list.append(f'{percentaje}%')
    return return_list


if __name__ == '__main__':
    NUM_INSTANCES = 'All'

    data = pd.read_csv(f'../Datasets/Suicide_Detection{NUM_INSTANCES}.csv')

    instances_lengths = []
    for text in data['text']:
        instances_lengths.append(len(text.split()))

    # Statistics
    print('Number of instance:', len(instances_lengths))
    print(f'\tMax words instance: {max(instances_lengths)} (instance {instances_lengths.index(max(instances_lengths))})')
    print('\tMin words instance:', min(instances_lengths))
    print('\tAverage words:', np.mean(instances_lengths))
    print('\tStandard deviation:', np.std(instances_lengths))

    distribution_in_hundreds = print_number_distribution(data)
    distribution_in_50s = number_distribution_in_Xs(data, 50)

    # Plots
    plot_real_distribution(instances_lengths)
    plot_less_600_words_distribution(instances_lengths)
    box_plot_less_400(instances_lengths)
    plot_with_class_distribution_less_2000_words(distribution_in_hundreds)
    plot_with_class_distribution_less_700_words(distribution_in_50s)