from typing import Callable, List

from src.genetic_algo.build_net.sample import Sample
from src.network import Network
from src.genetic_algo.train_net.simulator import Simulator as TrainSimulator

def correctness_ratio(simulator: TrainSimulator, sample: Sample, mutation_function: Callable[[List[int], float, float], None]) -> float:
    """
    Check the ratio of decoded words that appear in the corpus
    :param dec: List of decoded words
    :param corpus: List of Corpus words
    :return: float: ratio
    """
    network: Network = sample.network
    return simulator.resume_network(network, mutation_function)
