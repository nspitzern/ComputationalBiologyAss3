import copy
from typing import List
from random import randint

from src.genetic_algo.build_net.sample import Sample
from src.genetic_algo.selector import Selector


class Evolver:
    @staticmethod
    def single_crossover(samples: List[Sample], fitness_scores: List[float]) -> List[Sample]:
        # Choose 2 samples for crossover
        s1, s2 = Selector.choose_n_weighted_random(samples, fitness_scores, 2)
        
        i = randint(1, len(s1) - 2)
        j = randint(1, len(s2) - 2)
        
        cross1, cross2 = copy.deepcopy(s1), copy.deepcopy(s2)

        # Swap one layer
        dim_1, dim_2 = s1[i], s2[j]
        cross1[i], cross2[j] = dim_2, dim_1

        return [cross1, cross2]
    
    @staticmethod
    def one_point_crossover(samples: List[Sample], fitness_scores: List[float]) -> List[Sample]:
        # Choose 2 samples for crossover
        s1, s2 = Selector.choose_n_weighted_random(samples, fitness_scores, 2)
        min_len = min(len(s1), len(s2))

        if min_len < 4:
            return []
        
        i = randint(1, min_len - 2)

        cross1, cross2 = copy.deepcopy(s1), copy.deepcopy(s1)
        
        # Swap one part of the sample (multiple layers)
        dims_1_1, dims_2_1 = s1[:i], s2[:i]
        dims_1_2, dims_2_2 = s1[i:], s2[i:]
        cross1[:i], cross2[:i] = dims_2_1, dims_1_1
        cross1[i:], cross2[i:] = dims_1_2, dims_2_2

        return [cross1, cross2]
    
    @staticmethod
    def generate_crossovers(samples: List[Sample], fitness_scores: List[float], n: int) -> List[Sample]:
        """
        Generate n crossovers from given elite samples.

        Args:
            elite_samples (List[Sample]): samples on which to generate crossovers
            n (int): number of samples to generate

        Returns:
            List[Sample]: valid crossover samples
        """
        new_samples: List[Sample] = []
        samples_len = 0

        while samples_len < n:
            co = Evolver.single_crossover(samples, fitness_scores)

            for o in co:
                new_samples.append(o)

            samples_len += len(co)

        return new_samples
