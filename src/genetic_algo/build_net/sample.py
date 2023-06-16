from random import random, choice
from typing import List

from src.common.base_sample import BaseSample
from src.network import Network, Swish


class Sample(BaseSample):
    def __init__(self, sample_size: int, net_optional_sizes: List[int], mutation_threshold: float):
        self.__layer_dims = Sample.generate_layer_sizes(net_optional_sizes, sample_size, 0.25)
        self.__activations = [Swish] * (len(self.__layer_dims) - 1)
        self.__net_optional_sizes = net_optional_sizes
        self.__threshold = mutation_threshold
    
    @property
    def network(self):
        return Network(self.__layer_dims, self.__activations)
    
    def mutate(self) -> None:
        for i in range(len(self.__layer_dims)):
            if random() > self.__threshold:
                new_dim = choice(self.__net_optional_sizes)
                self.__layer_dims[i] = new_dim
    
    def __getitem__(self, item):
        return self.__layer_dims[item]
    
    def __setitem__(self, key, value):
        self.__layer_dims[key] = value

    def __len__(self):
        return len(self.__layer_dims)
    
    @staticmethod
    def generate_layer_sizes(net_optional_sizes: List[int], sample_size: int, threshold: float):
        # Add sample size as first size
        layer_dims: List[int] = []
        layer_dims.append(sample_size)

        # Randomly add layer sizes
        prob: float = random()
        while prob > threshold:
            selection = choice(net_optional_sizes)
            layer_dims.append(selection)
            prob = random()
        
        # Add wanted output size
        layer_dims.append(1)
        return layer_dims
