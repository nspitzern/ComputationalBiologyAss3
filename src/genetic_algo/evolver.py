import copy
from typing import List
from random import randint

from src.genetic_algo.sample import Sample
from src.genetic_algo.selector import Selector


class Evolver:
    @staticmethod
    def single_crossover(samples: List[Sample], fitness_scores: List[float]) -> List[Sample]:
        # Choose 2 samples for crossover
        s1, s2 = Selector.choose_n_weighted_random(samples, fitness_scores, 2)
        i = randint(0, s1.network_length - 1)
        
        cross1, cross2 = copy.deepcopy(s1), copy.deepcopy(s1)

        # Swap one layer
        layer_1, layer_2 = s1[i], s2[i]
        cross1[i], cross2[i] = layer_2, layer_1

        return [cross1, cross2]
    
    @staticmethod
    def one_point_crossover(samples: List[Sample], fitness_scores: List[float]) -> List[Sample]:
        # Choose 2 samples for crossover
        s1, s2 = Selector.choose_n_weighted_random(samples, fitness_scores, 2)
        i = randint(1, s1.network_length - 2)

        cross1, cross2 = copy.deepcopy(s1), copy.deepcopy(s1)
        
        # Swap one part of the sample (multiple layers)
        layer_1_1, layer_2_1 = s1[:i], s2[:i]
        layer_1_2, layer_2_2 = s1[i:], s2[i:]
        cross1[:i], cross2[:i] = layer_2_1, layer_1_1
        cross1[i:], cross2[i:] = layer_2_2, layer_1_2

        return [cross1, cross2]
