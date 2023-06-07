from random import shuffle

from src.data_parser import load_data_file, DataItem


class Dataset:
    def __init__(self, data_filepath: str, should_shuffle: bool = False):
        self.__items = load_data_file(data_filepath)
        self.__should_shuffle = should_shuffle

    def __len__(self) -> int:
        return len(self.__items)

    def __getitem__(self, item) -> DataItem:
        return self.__items[item]
    
    def __iter__(self):
        return (t for t in self.__items)

    def shuffle(self):
        shuffle(self.__items)
