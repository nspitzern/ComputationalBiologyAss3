from random import random
from typing import Union

import numpy as np

from src.network import Network, save_network


class Sample:
    def __init__(self, network: Network, mutation_magnitude: float):
        self.__network: Network = network
        self.__magnitude = mutation_magnitude

    def mutate_additive(self, threshold: float):
        for l in self.__network.layers:
            if random() > threshold:
                x = 1 + np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
                l.weights *= x
    
    def mutate_multiplicative(self, threshold: float):
        for l in self.__network.layers:
            if random() > threshold:
                x = 1 + np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
                l.weights *= x
    
    def save(self, filepath):
        save_network(filepath, self.__network)
    
    @property
    def network_length(self):
        return len(self.__network)
    
    def __getitem__(self, item):
        return self.__network[item]
    
    def __setitem__(self, key, value):
        self.__network[key] = value
    
    def __call__(self, x: np.ndarray) -> Union[np.ndarray, int]:
        return self.__network(x)
