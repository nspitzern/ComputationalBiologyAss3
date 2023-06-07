import copy
from typing import List
from random import randint

from src.genetic_algo.sample import Sample
from src.genetic_algo.selector import Selector
from src.network import Network


class Evolver:
    @staticmethod
    def one_point_crossover(samples: List[Sample], fitness_scores: List[float]) -> List[Sample]:
        # Choose 2 samples for crossover
        s1, s2 = Selector.choose_n_weighted_random(samples, fitness_scores, 2)
        i = randint(0, len(s1) - 1)

        cross1, cross2 = copy.deepcopy(s1), copy.deepcopy(s1)
        layer_1, layer_2 = s1[i], s2[i]
        cross1[i], cross2[i] = layer_2, layer_1

        return [cross1, cross2]
