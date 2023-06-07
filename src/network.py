from typing import List
from abc import ABC
import json

import numpy as np


class ActivationFunction(ABC):
    def __call__(self, x: np.ndarray):
        raise NotImplemented

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


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
        self.weights: np.ndarray = np.random.random(size=(input_dim, output_dim))
        self.shape = self.weights.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def to_JSON(self):
        return self.weights.tolist()


class Network:
    def __init__(self, layers_dim: List[int], activations: List[ActivationFunction]):
        self._layers = []
        self._activations = []

        for dim_idx in range(len(layers_dim) - 1):
            input_l_dim = layers_dim[dim_idx]
            output_l_dim = layers_dim[dim_idx + 1]
            self._layers.append(Layer(input_dim=input_l_dim, output_dim=output_l_dim))

        for activation in activations:
            self._activations.append(activation())

        self._activations.append(Sigmoid())

    def forward(self, x: np.ndarray) -> np.ndarray:
        for f, a in zip(self._layers, self._activations):
            x = f(x)
            x = a(x)

        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def save_network(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'weight': [l.to_JSON() for l in self._layers],
                'activation': [a.to_JSON() for a in self._activations]
            }, f)

    def load_network(self, filepath: str):
        pass
