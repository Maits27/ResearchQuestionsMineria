from loadSaveData import loadRAW
import matplotlib.pyplot as plt
import numpy as np


def incrementValue(dict, length, clase):
    if length in dict:
        if clase in dict[length]: dict[length][clase] += 1 
        else: dict[length][clase] = 1
    else: 
        dict[length] = {clase: 1}


def plot_real_distribution(instances_lengths):
    plt.hist(instances_lengths)
    plt.show() 


def print_number_distribution(data):
    distribution = {}
    for index in data.index:
        length = len(data['text'][index])
        resto = length % 100
        cuantoHasta100 = 100 - resto
        divisorDe100Siguiente = length + cuantoHasta100
        incrementValue(distribution, divisorDe100Siguiente, data['class'][index])


    for key in range(100, 2000, 100):
        print(f'<={key}: suicide: {distribution[key]["suicide"]}; non-suicide: {distribution[key]["non-suicide"]}')
    return distribution


def plot_less_1000_words_distribution(instances_lengths):
    less_1000_instances_length = []  
    for length in instances_lengths:
        if length <= 1000: less_1000_instances_length.append(length)

    plt.hist(less_1000_instances_length)
    plt.show() 


def plot_less_600_words_distribution(instances_lengths):
    less_600_instances_length = []  
    for length in instances_lengths:
        if length <= 600: less_600_instances_length.append(length)

    plt.hist(less_600_instances_length, bins=range(0, 601, 1), label='Number of words per instance')
    plt.show() 


def box_plot(instances_lengths):
    data = plt.boxplot(instances_lengths)
    plt.show()
    outliers = data['fliers'][0].get_data()[1]
    print('Min outlier', min(outliers))


def box_plot_less_1000(instances_lengths):
    less_1000_instances_length = []  
    for length in instances_lengths:
        if length <= 1000: less_1000_instances_length.append(length)

    plt.boxplot(less_1000_instances_length)
    plt.title('Without >1000 length instances')
    plt.show()


def box_plot_less_400(instances_lengths):
    less_400_instances_length = []  
    for length in instances_lengths:
        if length <= 400: less_400_instances_length.append(length)

    plt.boxplot(less_400_instances_length)
    plt.title('Without >400 length instances')
    plt.show()


def plot_with_class_distribution(distribution):
    groups = list(range(100, 2000, 100))
    group_count = {
        'suicide': [distribution[group]['suicide'] for group in groups],
        'non-suicide': [distribution[group]['non-suicide'] for group in groups]
    }

    width = 75

    fig, ax = plt.subplots()
    bottom = np.zeros(len(groups))

    for clase, counts in group_count.items():
        ax.bar(groups, counts, width, label=clase, bottom=bottom)
        bottom += counts

    ax.set_title("Class distribution on different sentence lengths")
    ax.legend(loc="upper right")
    
    plt.xticks(list(range(100, 2000, 100)))
    plt.xlabel('Sentence word length')
    plt.show()


if __name__ == '__main__':
    data = loadRAW('../Datasets/Suicide_Detection10000.csv')

    instances_lengths = []
    for text in data['text']:
        instances_lengths.append(len(text.split()))

    print('Max words instance:', max(instances_lengths))
    print('Min words instance:', min(instances_lengths))
    print('Average words:', np.mean(instances_lengths))
    print('Standard deviation:', np.std(instances_lengths))

    distribution = print_number_distribution(data)
    plot_real_distribution(instances_lengths)
    plot_less_1000_words_distribution(instances_lengths)
    plot_less_600_words_distribution(instances_lengths)
    box_plot(instances_lengths)
    box_plot_less_1000(instances_lengths)
    box_plot_less_400(instances_lengths)
    plot_with_class_distribution(distribution)