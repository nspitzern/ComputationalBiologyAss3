from matplotlib import pyplot as plt
from src.dataset import Dataset
import numpy as np

def subplot(num, data):
    plt.subplot(2, 1, num)
    plt.bar([x for x in range(1, 17)], data)
    # for i, v in enumerate(data):
    #     plt.text(i + 1, v+25, "%d" %v, ha="center")
    plt.xlabel(f"Number of {num - 1}'s")
    plt.ylabel('Count')

def plot(data_filename: str, data1, data2, wanted_tag):
    plt.figure(figsize=(8, 6), dpi=70)
    plt.suptitle(f'{data_filename} count for tag {wanted_tag}', fontsize=14)
    subplot(1, data1)
    subplot(2, data2)
    plt.savefig(f'plot_{data_filename}_{wanted_tag}.png', format='png')
    plt.cla()

if __name__ == '__main__':
    filename = 'nn1'
    dataset = Dataset(f'{filename}.txt')

    count = 0
    length = len(dataset[0].sample)
    counts_0 = np.array([0] * length)
    counts_1 = np.array([0] * length)
    wanted_tag = 0

    for arr, tags in dataset:
        for s, t in zip(arr, tags):
            count += 1
            if t == wanted_tag:
                temp = sum(s)
                counts_1[temp - 1] += 1
                temp = length - sum(s)
                counts_0[temp - 1] += 1
    
    plot(filename, counts_0, counts_1, wanted_tag)
