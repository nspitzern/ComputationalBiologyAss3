from typing import Callable, Union
import numpy as np

from src.network import Network, save_network


class Sample:
    def __init__(self, network: Network, mutation_function: Callable[[Network, float, float], None], 
                 mutation_threshold: float, mutation_magnitude: float):
        self.__network: Network = network
        self.__mutation_function = mutation_function
        self.__threshold = mutation_threshold
        self.__magnitude = mutation_magnitude
    
    def mutate(self):
        self.__mutation_function(self.__network, self.__threshold, self.__magnitude)

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
