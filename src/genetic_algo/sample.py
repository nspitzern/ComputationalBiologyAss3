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
            weights_to_change = np.random.random(l.shape) >= threshold
            scale = np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
            l.weights += weights_to_change * scale
    
    def mutate_multiplicative(self, threshold: float):
        for l in self.__network.layers:
            weights_to_change = np.random.random(l.shape) >= threshold
            scale = 1 + np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
            l.weights = np.where(weights_to_change, l.weights * scale , l.weights)
    
    def mutate_random(self, threshold: float):
        for l in self.__network.layers:
            if random() > threshold:
                l.weights = np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)

    def save(self, filepath: str):
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
