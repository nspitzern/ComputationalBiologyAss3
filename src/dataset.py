from typing import Tuple
from random import shuffle

from numpy import ndarray

from src.data_parser import load_data_file


class Dataset:
    def __init__(self, data_filepath: str, should_shuffle: bool = False):
        self.__samples, self.__labels = load_data_file(data_filepath)
        self.__should_shuffle = should_shuffle

    def __len__(self) -> int:
        return len(self.__samples)

    def __getitem__(self, item) -> Tuple[ndarray, int]:
        return self.__samples[item], self.__labels[item]

    def shuffle(self):
        temp = list(zip(self.__samples, self.__labels))
        shuffle(temp)
        self.__samples, self.__labels = zip(*temp)
