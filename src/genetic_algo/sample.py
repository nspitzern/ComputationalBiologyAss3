from random import random
import numpy as np

from src.network import DynamicNetwork


class Sample:
    def __init__(self, network: DynamicNetwork, mutation_magnitude: float):
        self.__network: DynamicNetwork = network
        self.__magnitude = mutation_magnitude

    def mutate(self, threshold: float):
        for l in self.__network.layers:
            if random() > threshold:
                x = np.random.uniform(low=-self.__magnitude, high=self.__magnitude, size=l.shape)
                l.weights += x
