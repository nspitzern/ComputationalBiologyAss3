from typing import List
from math import floor
from random import choices, sample

from src.genetic_algo.sample import Sample


class Selector:
    @staticmethod
    def select_elite(samples, fitness_scores, percentile: float) -> List[Sample]:
        """
        Returns a list of top percentile samples.
        
        :param samples: List of samples
        :param fitness_scores: Sample fitness scores
        :param percentile: Which percentile to keep
        :return:
        """
        # Calculate starting index from which to select
        start_index = floor(percentile * len(fitness_scores))

        # Sort the samples by fitness
        sorted_samples, fitness_scores = zip(*sorted(zip(samples, fitness_scores), key=lambda x: x[1]))
        return sorted_samples[start_index:]

    @staticmethod
    def choose_n_random(arr: List, n: int) -> List[int]:
        return sample(range(len(arr)), n)
    
    @staticmethod
    def choose_n_weighted_random(arr: List, weights: List[float], n: int) -> List[Sample]:
        return choices(arr, weights, k=n)
