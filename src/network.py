from enum import StrEnum
from typing import List
from abc import ABC
import json
import numpy as np


class ActivationFunctionType(StrEnum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'

    @staticmethod
    def get_activation(a_type: str):
        activations = {
            'relu': ReLU,
            'sigmoid': Sigmoid
        }

        return activations[a_type]


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


class StepFunction:
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray, threshold: float = 0.5) -> int:
        return 1 if x > threshold else 0


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
        self.__step_func = StepFunction()

    @property
    def layers(self):
        return self.__layers

    @property
    def activations(self):
        return self.__activations

    def forward(self, x: np.ndarray) -> np.ndarray:
        for f, a in zip(self.__layers, self.__activations):
            x = f(x)
            x = a(x)

        return self.__step_func(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def __getitem__(self, item):
        return self.__layers[item]
    
    def __setitem__(self, key, value):
        self.__layers[key] = value
    
    def __len__(self):
        return len(self.__layers)


def save_network(filepath: str, network: Network):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'weights': [l.to_JSON() for l in network.layers],
            'activations': [str(a) for a in network.activations]
        }, f)


def load_network(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        net_structure = json.load(f)

        activations = [ActivationFunctionType.get_activation(a_type) for a_type in net_structure['activations']]
        weights = [weight for weight in net_structure['weights']]

        dims_list = []
        for l_weight in weights:
            weight = np.array(l_weight)
            dims_list.append(weight.shape[0])

        dims_list.append(weight.shape[-1])

        net = Network(layers_dim=dims_list, activations=activations)

        for i, l_weight in enumerate(weights):
            weight = np.array(l_weight)
            net.layers[i].weights = weight

        return net
