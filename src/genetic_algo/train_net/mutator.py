from random import random
import numpy as np

from src.network import Network


class Mutator:
    @staticmethod
    def mutate_additive(network: Network, threshold: float, magnitude: float):
        for l in network.layers:
            weights_to_change = np.random.random(l.shape) >= threshold
            scale = np.random.uniform(low=-magnitude, high=magnitude, size=l.shape)
            l.weights += weights_to_change * scale
    
    @staticmethod
    def mutate_multiplicative(network: Network, threshold: float, magnitude: float):
        for l in network.layers:
            weights_to_change = np.random.random(l.shape) >= threshold
            scale = 1 + np.random.uniform(low=-magnitude, high=magnitude, size=l.shape)
            l.weights = np.where(weights_to_change, l.weights * scale , l.weights)
    
    @staticmethod
    def mutate_random(network: Network, threshold: float, magnitude: float):
        for l in network.layers:
            if random() > threshold:
                l.weights = np.random.uniform(low=-magnitude, high=magnitude, size=l.shape)
