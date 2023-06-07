from enum import StrEnum
from typing import List
from abc import ABC
import json
import numpy as np

class ActivationFunctionType(StrEnum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'


class ActivationFunction(ABC):
    def __init__(self, type: ActivationFunctionType) -> None:
        super().__init__()
        self.__type = type

    def __call__(self, x: np.ndarray):
        raise NotImplemented
    
    def __str__(self) -> str:
        return self.__type


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__(ActivationFunctionType.RELU)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__(ActivationFunctionType.SIGMOID)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 - np.e ** -x)


class BatchNorm:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        raise NotImplemented


class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights: np.ndarray = np.random.random(size=(input_dim, output_dim))
        self.shape = self.weights.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def to_JSON(self):
        return self.weights.tolist()


class Network:
    def __init__(self, layers_dim: List[int], activations: List[ActivationFunction]):
        self.__layers = []
        self.__activations = []

        for dim_idx in range(len(layers_dim) - 1):
            input_l_dim = layers_dim[dim_idx]
            output_l_dim = layers_dim[dim_idx + 1]
            self.__layers.append(Layer(input_dim=input_l_dim, output_dim=output_l_dim))

        for activation in activations:
            self.__activations.append(activation())

        self.__activations.append(Sigmoid())

    @property
    def layers(self):
        return self.__layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for f, a in zip(self.__layers, self.__activations):
            x = f(x)
            x = a(x)

        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def __getitem__(self, item):
        return self.__layers[item]
    
    def __setitem__(self, key, value):
        self.__layers[key] = value
    
    def __len__(self):
        return len(self.__layers)

    def save_network(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'weight': [l.to_JSON() for l in self.__layers],
                'activation': [str(a) for a in self.__activations]
            }, f)

    def load_network(self, filepath: str):
        pass
