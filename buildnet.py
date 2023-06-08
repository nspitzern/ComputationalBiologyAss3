from src.dataset import Dataset
from src.genetic_algo.simulator import SimulationArgs, Simulator
from src.genetic_algo.strategy import GeneticAlgorithmType


if __name__ == '__main__':
    dataset: Dataset = Dataset('nn0.txt')
    args: SimulationArgs = SimulationArgs(0.97, elite_percentile=0.9, mutation_percentage=0.2, mutation_threshold=0.5)
    simulator = Simulator(GeneticAlgorithmType.REGULAR, 100, dataset, args)
    simulator.run()