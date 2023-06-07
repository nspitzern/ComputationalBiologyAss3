from random import random
import numpy as np

from src.network import Network


class Sample:
    def __init__(self, network: Network, mutation_magnitude: float):
        self.__network: Network = network
        self.__magnitude = mutation_magnitude

    def mutate(self, threshold: float):
        for l in self.__network.layers:
            if random() > threshold:
                x = np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
                l.weights += x
    
    def __getitem__(self, item):
        return self.__network[item]
    
    def __setitem__(self, key, value):
        self.__network[key] = value
    
    def __len__(self):
        return len(self.__network)
