from enum import StrEnum
from typing import List, Union
from abc import ABC
import json
import numpy as np


class ActivationFunctionType(StrEnum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SWISH = 'swish'

    @staticmethod
    def get_activation(a_type: str):
        activations = {
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'swish': Swish
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
        return 1 / (1 + np.e ** -x)


class Tanh(ActivationFunction):
    def __init__(self):
        super().__init__(ActivationFunctionType.TANH)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        e_x = np.e ** x
        e_minus_x = np.e ** -x
        return (e_x - e_minus_x) / (e_x + e_minus_x)


class Swish(ActivationFunction):
    def __init__(self):
        super(Swish, self).__init__(ActivationFunctionType.SWISH)
        self.sigmoid = Sigmoid()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.sigmoid(x)


class StepFunction:
    def __init__(self):
        super().__init__()

    def __call__(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.array(x > threshold, dtype=np.int8)


class BatchNormLayer:
    def __init__(self, dims: int) -> None:
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initilizaded at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x

    def __call__(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.mean((x - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += 10e-3
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma * self.standard_x + self.bias


class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights: np.ndarray = self.__xavier_init(input_dim, output_dim)
        # self.weights: np.ndarray = np.random.random(size=(input_dim, output_dim))
        self.shape = self.weights.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights

    def to_JSON(self):
        return self.weights.tolist()

    def __xavier_init(self, in_dim: int, out_dim: int) -> np.ndarray:
        return np.random.uniform(low=-1, high=1, size=(in_dim, out_dim)) * np.sqrt(6. / (in_dim + out_dim))


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

        self.__sigmoid = Sigmoid()
        self.__step_func = StepFunction()

    @property
    def layers(self):
        return self.__layers

    @property
    def activations(self):
        return self.__activations

    def forward(self, x: np.ndarray) -> Union[np.ndarray, int]:
        for f, a in zip(self.__layers, self.__activations):
            x = f(x)
            x = a(x)

        return self.__step_func(self.__sigmoid(x))

    def __call__(self, x: np.ndarray) -> Union[np.ndarray, int]:
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
