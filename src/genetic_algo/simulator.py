from typing import List
from src.dataset import Dataset

from src.genetic_algo.evolver import Evolver
from src.genetic_algo.sample import Sample
from src.genetic_algo.selector import Selector
from src.genetic_algo.strategy import GeneticAlgorithmType
from src.network import Network, ReLU


class MutationArgs:
    def __init__(self, mutation_percentage: float, mutation_threshold: float, mutation_magnitude:float) -> None:
        self.mutation_percentage = mutation_percentage
        self.mutation_threshold = mutation_threshold
        self.mutation_magnitude = mutation_magnitude


class SimulationArgs:
    def __init__(self, fitness_goal: float, elite_percentile: float, mutation_percentage: float, 
                 mutation_threshold: float, mutation_magnitude: float) -> None:
        self.fitness_goal = fitness_goal
        self.elite_percentile = elite_percentile
        self.mutation = MutationArgs(mutation_percentage, mutation_threshold, mutation_magnitude)


class Simulator:
    def __init__(self, algo_type: GeneticAlgorithmType, num_samples: int, dataset: Dataset, simulation_args: SimulationArgs) -> None:
        self.__args: SimulationArgs = simulation_args
        self.__fitness_goal: float = simulation_args.fitness_goal
        self.algo_type = algo_type
        self.__strategy = GeneticAlgorithmType.get_strategy(algo_type, dataset, self.__args.mutation.mutation_threshold)
        self.__num_samples = num_samples
        self.__elite_percentile = simulation_args.elite_percentile

    def __should_run(self, fitness_scores: List[float]):
        return all([f < 0.97 for f in fitness_scores])

    def __generate_crossovers(self, samples: List[Sample], fitness_scores: List[float], n: int) -> List[Sample]:
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
            co = Evolver.one_point_crossover(samples, fitness_scores)

            for o in co:
                new_samples.append(o)

            samples_len += len(co)

        return new_samples

    def __save(self):
        pass

    def __step(self, step: int, samples: List[Sample], fitness_scores: List[float]):
        # Selection
        elite_samples = Selector.select_elite(samples, fitness_scores, self.__elite_percentile)
        
        # Crossover
        samples = self.__generate_crossovers(samples, fitness_scores, self.__num_samples - len(elite_samples))
        
        # Mutation
        mutation_amount = int(len(samples) * self.__args.mutation.mutation_percentage)

        for i in Selector.choose_n_random(samples, mutation_amount):
            samples[i].mutate(self.__args.mutation.mutation_threshold)

        samples.extend(elite_samples)

        # Compute fitness
        fitness_scores = self.__strategy.fitness(samples)

        return samples, fitness_scores

    def run(self):
        step = 0

        layer_dims, activations = [16, 32, 16, 1], [ReLU, ReLU, ReLU]
        mutation_magnitude = self.__args.mutation.mutation_magnitude
 
        # Generate initial population
        samples: List[Sample] = [Sample(Network(layer_dims, activations), mutation_magnitude) for _ in range(self.__num_samples)]
        
        # Compute fitness
        fitness_scores = self.__strategy.fitness(samples)
        print("Max fitness:", max(fitness_scores))
        
        print(f'Mutation rate: {self.__args.mutation.mutation_percentage}')

        while self.__should_run(fitness_scores):
            print("activating")
            step_func = lambda s, f: self.__step(step, s, f)
            samples, fitness_scores = self.__strategy.activate(step_func, samples, fitness_scores)
            print("Max fitness:", max(fitness_scores))
            step += 1

        return fitness_scores, samples
