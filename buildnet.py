from src.network import ReLU
from src.dataset import Dataset
from src.genetic_algo.simulator import SimulationArgs, Simulator
from src.genetic_algo.strategy import GeneticAlgorithmType


if __name__ == '__main__':
    sample_size = 100
    dataset: Dataset = Dataset('nn0.txt', batch_size=256)
    layer_dims, activations = [len(dataset[0].sample), 32, 16, 1], [ReLU, ReLU, ReLU]
    args: SimulationArgs = SimulationArgs(1, elite_percentile=0.9, mutation_percentage=0.5, 
                                          mutation_threshold=0.3, mutation_magnitude=0.05)
    simulator = Simulator(GeneticAlgorithmType.REGULAR, sample_size, dataset, args)
    simulator.run(layer_dims, activations)
