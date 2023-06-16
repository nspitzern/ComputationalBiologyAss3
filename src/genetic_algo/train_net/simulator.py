import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Callable, List, Tuple

from src.common.simulation_history import SimulationHistory
from src.dataset import Dataset, split_train_test
from src.genetic_algo.train_net.evolver import Evolver
from src.genetic_algo.train_net.sample import Sample
from src.genetic_algo.selector import Selector
from src.genetic_algo.train_net.strategy import GeneticAlgorithmType
from src.network import ActivationFunction, Network, load_network


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
    def __init__(self, algo_type: GeneticAlgorithmType, num_samples: int, dataset: Dataset, 
                 output_dir_path: str, simulation_args: SimulationArgs, train_ratio: int = 0.7, max_steps: int = -1, silent: bool = False, plot: bool = False) -> None:
        self.__args: SimulationArgs = simulation_args
        self.__fitness_goal: float = simulation_args.fitness_goal
        self.__train_dataset, self.__test_dataset = split_train_test(dataset, train_ratio=train_ratio)
        self.algo_type = algo_type
        self.__strategy = GeneticAlgorithmType.get_strategy(algo_type)
        self.__num_samples = num_samples
        self.__output_dir_path = output_dir_path
        self.__elite_percentile = simulation_args.elite_percentile
        self.__max_steps = max_steps
        self.__silent = silent
        self.__plot = plot

    def __log(self, message: str):
        if self.__silent:
            return
        
        print(message)

    def __should_run(self, step: int, fitness_scores: List[float]):
        if self.__max_steps != -1 and step >= self.__max_steps:
            return False
        return all([f < self.__fitness_goal for f in fitness_scores])

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

    def __save(self, samples: List[Sample], fitness_scores: List[float], filename: str):
        if not self.__silent:
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.__output_dir_path, exist_ok=True)

        # Save plot
        plt.savefig(os.path.join(self.__output_dir_path, f'plot_{filename}.png'), format='png')

        # Save weights of best sample
        i = np.argmax(fitness_scores)
        best: Sample = samples[i]
        best.save(os.path.join(self.__output_dir_path, f'net_{filename}.json'))

        # Save simulator arguments
        filename = os.path.join(self.__output_dir_path, f'data_{filename}.txt')
        with open(filename, '+wt', encoding='utf-8') as f:
            f.write(f'strategy: {GeneticAlgorithmType.map_to_str(self.algo_type)}{os.linesep}')
            f.write(f'mutation function: {str(best._Sample__mutation_function.__name__)}{os.linesep}')
            f.write(f'sample size: {self.__num_samples}{os.linesep}')
            f.write(f'fitness score: {fitness_scores[i] * 100:.3f}{os.linesep}')
            f.write(f'fitness calls: {self.__strategy.fitness_calls}{os.linesep}')
            f.write(f'elite percentage: {self.__elite_percentile}{os.linesep}')
            f.write(f'mutation percentage: {self.__args.mutation.mutation_percentage}{os.linesep}')
            f.write(f'minimum mutation threshold: {self.__args.mutation.mutation_threshold}{os.linesep}')
            f.write(f'minimum mutation magnitude: {self.__args.mutation.mutation_magnitude}{os.linesep}')

    def __plot_current(self, history: SimulationHistory):
        best_overall = max(history.best)
        plt.title(f'Best fitness: {best_overall}\nMethod: {GeneticAlgorithmType.map_to_str(self.algo_type)}, Population Size: {self.__num_samples},\n Fitness Calls: {self.__strategy.fitness_calls}, Mutation Ratio: {self.__args.mutation.mutation_percentage * 100}%')
        plt.plot(history.worst, label='Worst Fitness')
        plt.plot(history.average, label='Avg Fitness')
        plt.plot(history.best, label='Best Fitness')
        plt.xlabel('Generation number')
        plt.ylabel('Fitness Score %')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        if self.__plot:
            plt.draw()
            plt.pause(0.01)
    
    def __add_current_iteration_data(self, fitness_scores: List[float], 
                                     history: SimulationHistory):
        worst: float = min(fitness_scores) * 100
        average: float = statistics.mean(fitness_scores) * 100
        best: float = max(fitness_scores) * 100
        history.add(worst, average, best)

    def __step(self, samples: List[Sample], fitness_scores: List[float]):
        # Selection
        elite_samples = Selector.select_elite(samples, fitness_scores, self.__elite_percentile)
        
        # Crossover
        samples = self.__generate_crossovers(samples, fitness_scores, self.__num_samples - len(elite_samples))
        
        # Mutation
        mutation_amount = int(len(samples) * self.__args.mutation.mutation_percentage)

        for i in Selector.choose_n_random(samples, mutation_amount):
            samples[i].mutate()

        samples.extend(elite_samples)

        # Compute fitness
        fitness_scores = self.__strategy.fitness(samples, self.__train_dataset)

        return samples, fitness_scores
    
    def __run_logic(self, samples: List[Sample]) -> Tuple[List[float], List[Sample]]:
        history: SimulationHistory = SimulationHistory()
        step = 0
        best_score = 0
        filename: str = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        plt.figure(figsize=(8, 6), dpi=70)
        plt.title(f'Initial mutation percentage: {self.__args.mutation.mutation_percentage * 100}%, elite percentile: {self.__elite_percentile * 100}%')
 
        # Compute fitness
        train_fitness_scores = self.__strategy.fitness(samples, self.__train_dataset)
        test_fitness_scores = self.__strategy.fitness(samples, self.__test_dataset)
        self.__log(f'Max fitness: {max(test_fitness_scores)}')
        
        self.__log(f'Mutation rate: {self.__args.mutation.mutation_percentage}')

        self.__add_current_iteration_data(test_fitness_scores, history)
        self.__plot_current(history)
        plt.cla()

        while self.__should_run(step, test_fitness_scores):
            self.__log(f'step {step}')
            samples, train_fitness_scores = self.__strategy.activate(self.__step, samples, self.__train_dataset, train_fitness_scores)
            test_fitness_scores = self.__strategy.fitness(samples, self.__test_dataset)
            self.__log(f'Max fitness: {max(test_fitness_scores)}')

            self.__add_current_iteration_data(test_fitness_scores, history)
            self.__plot_current(history)

            # Save best results until now
            if best_score < max(test_fitness_scores):
                self.__save(samples, test_fitness_scores, filename)

            # Shuffle train dataset so we don't overfit
            self.__train_dataset.shuffle()
            
            plt.cla()

            step += 1

        self.__plot_current(history)
        self.__save(samples, test_fitness_scores, filename)
        return test_fitness_scores, samples

    def run(self, layer_dims: List[int], activations: List[ActivationFunction], 
            mutation_function: Callable[[Network, float, float], None]) -> Tuple[List[float], List[Sample]]:
        mutation_threshold = self.__args.mutation.mutation_threshold
        mutation_magnitude = self.__args.mutation.mutation_magnitude

        # Generate initial population
        samples: List[Sample] = [Sample(Network(layer_dims, activations), mutation_function, mutation_threshold, mutation_magnitude) for _ in range(self.__num_samples)]
        return self.__run_logic(samples)
    
    def resume(self, filename: str, mutation_function: Callable[[Network, float, float], None]) -> Tuple[List[float], List[Sample]]:
        # Load population from a saved network
        network = load_network(filename)
        return self.resume_network(network, mutation_function)
    
    def resume_network(self, network: Network, mutation_function: Callable[[Network, float, float], None]) -> Tuple[List[float], List[Sample]]:
        # Load population from a given network
        mutation_threshold = self.__args.mutation.mutation_threshold
        mutation_magnitude = self.__args.mutation.mutation_magnitude
        samples: List[Sample] = [Sample(network, mutation_function, mutation_threshold, mutation_magnitude) for _ in range(self.__num_samples)]
        return self.__run_logic(samples)
