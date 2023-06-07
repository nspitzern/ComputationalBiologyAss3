from typing import List
from abc import ABC

import numpy as np


class ActivationFunction(ABC):
    def __call__(self, x: np.ndarray):
        raise NotImplemented


class ReLU(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sigmoid(ActivationFunction):
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 - np.e ** -x)


class BatchNorm:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        raise NotImplemented


class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.random(size=(input_dim, output_dim))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights


class Network:
    def __init__(self, layers_dim: List[int], activation: ActivationFunction):
        self.__layers = []

        for dim_idx in range(len(layers_dim) - 1):
            input_l_dim = layers_dim[dim_idx]
            output_l_dim = layers_dim[dim_idx + 1]
            self.__layers.append(Layer(input_dim=input_l_dim, output_dim=output_l_dim))
            self.__layers.append(activation())

        self.__layers[-1] = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        for f in self.__layers:
            x = f(x)

        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def save_network(self):
        pass

    def load_network(self, filepath: str):
        pass
