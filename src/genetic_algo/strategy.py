import copy
from enum import IntEnum
from typing import Callable, List, Tuple
from src.dataset import Dataset

from src.genetic_algo.fitness import correctness_ratio
from src.genetic_algo.sample import Sample


class GeneticAlgorithmType(IntEnum):
    REGULAR = 0
    DARWIN = 1
    LAMARCK = 2

    @staticmethod
    def get_strategy(strategy: int):
        if strategy == 1:
            return DarwinStrategy()
        if strategy == 2:
            return LamarckStrategy()
        return RegularStrategy()

    @staticmethod
    def map_to_str(strategy: int) -> str:
        m = {
            GeneticAlgorithmType.REGULAR: 'REGULAR',
            GeneticAlgorithmType.DARWIN: 'DARWIN',
            GeneticAlgorithmType.LAMARCK: 'LAMARCK',
        }

        return m.get(strategy, '')


class BaseStrategy:
    def __init__(self) -> None:
        self.fitness_calls: int = 0

    def fitness(self, samples: List[Sample], dataset: Dataset) -> List[float]:
        self.fitness_calls += len(samples)

        fitness_scores = [correctness_ratio(s, dataset) for s in samples]
        return fitness_scores


class OptimizationStrategy(BaseStrategy):
    def optimize(self, samples: List[Sample], dataset: Dataset, fitness_scores: List[float]) -> List[Sample]:
        optimized: List[Sample] = list()
        prev_fitness = 0

        for s, f in zip(samples, fitness_scores):
            new_sample = s
            temp: Sample = copy.deepcopy(s)
            for _ in range(10):
                temp.mutate()

            new_fitness = self.fitness([temp], dataset)[0]
            # Accept mutation only if it is better
            if new_fitness >= f and new_fitness > prev_fitness:
                new_sample = temp
                prev_fitness = new_fitness
            
            optimized.append(new_sample)
        
        return optimized


class RegularStrategy(BaseStrategy):
    def activate(self, step_func: Callable[[List[Sample], List[float]], Tuple[List[Sample], List[float]]], 
                 samples: List[Sample], dataset: Dataset, fitness_scores: List[float]) -> Tuple[List[Sample], List[float]]:
        return step_func(samples, fitness_scores)


class DarwinStrategy(OptimizationStrategy):
    def activate(self, step_func: Callable[[List[Sample], List[float]], Tuple[List[Sample], List[float]]], 
                 samples: List[Sample], dataset: Dataset, fitness_scores: List[float]) -> Tuple[List[Sample], List[float]]:
        optimized_samples = self.optimize(samples, dataset, fitness_scores)
        optimized_fitness = self.fitness(optimized_samples, dataset)
        samples, fitness_scores = step_func(samples, optimized_fitness)
        return samples, fitness_scores


class LamarckStrategy(OptimizationStrategy):
    def activate(self, step_func: Callable[[List[Sample], List[float]], Tuple[List[Sample], List[float]]], 
                 samples: List[Sample], dataset: Dataset, fitness_scores: List[float]) -> Tuple[List[Sample], List[float]]:
        optimized_samples = self.optimize(samples, dataset, fitness_scores)
        optimized_fitness = self.fitness(optimized_samples, dataset)
        samples, fitness_scores = step_func(optimized_samples, optimized_fitness)
        return samples, fitness_scores
