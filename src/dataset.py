from random import shuffle, sample
from typing import List, Tuple

import numpy as np

from src.data_parser import load_data_file, DataItem


class Dataset:
    def __init__(self, data_filepath: str = '', data_items: List[DataItem] = None, batch_size :int = 128, should_shuffle: bool = False):
        if data_items:
            self.__items = data_items
        elif data_filepath != '':
            self.__items = load_data_file(data_filepath)
        else:
            raise ValueError('Dataset not provided with data to load')
        self.__should_shuffle = should_shuffle
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.__items)

    def __getitem__(self, item) -> DataItem:
        return self.__items[item]

    def __iter__(self):
        num_batches = len(self.__items) // self.batch_size
        reminder = len(self.__items) % self.batch_size

        for i in range(num_batches):
            batch_s = [item.sample for item in self.__items[i * self.batch_size: i * self.batch_size + self.batch_size]]
            batch_l = [item.label for item in self.__items[i * self.batch_size: i * self.batch_size + self.batch_size]]
            yield np.vstack(batch_s), np.vstack(batch_l)

        batch_s = [item.sample for item in self.__items[reminder:]]
        batch_l = [item.label for item in self.__items[reminder:]]

        return np.vstack(batch_s), np.vstack(batch_l)

    # def __iter__(self):
    #     return (t for t in self.__items)

    def shuffle(self):
        shuffle(self.__items)


def split_train_test(items: Dataset, train_ratio: float = 0.7) -> Tuple[Dataset, Dataset]:
    num_items = len(items)
    train_idxs = sample(list(range(num_items)), int(train_ratio * num_items))
    test_idxs = set(range(num_items)) - set(train_idxs)

    train_items = [items[i] for i in train_idxs]
    test_items = [items[i] for i in test_idxs]

    return Dataset(data_items=train_items), Dataset(data_items=test_items)
