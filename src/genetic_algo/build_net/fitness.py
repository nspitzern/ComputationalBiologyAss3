from typing import Callable, List

import numpy as np

from src.genetic_algo.build_net.sample import Sample
from src.network import Network
from src.genetic_algo.train_net.simulator import Simulator as TrainSimulator

def correctness_ratio(simulator: TrainSimulator, sample: Sample, mutation_function: Callable[[List[int], float, float], None]) -> float:
    """
    Get the correctness ration of the sample network after training the genetic algorithm
    Args:
        simulator (TrainSimulator): simulator to run
        sample (Sample): sample network to test
        mutation_function (Callable[[List[int], float, float], None]): mutation function to apply

    Returns:
        float: ratio
    """
    network: Network = sample.network
    fitness_scores, samples = simulator.resume_network(network, mutation_function)
    i = np.argmax(fitness_scores)
    if sample.best_score < fitness_scores[i]:
        sample.best_sample = samples[i]
        sample.best_score = fitness_scores[i]
    return sample.best_score
