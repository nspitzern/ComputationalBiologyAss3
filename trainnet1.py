import sys
from src.network import Swish
from src.dataset import Dataset
from src.genetic_algo.train_net.simulator import SimulationArgs, Simulator
from src.genetic_algo.train_net.strategy import GeneticAlgorithmType
from src.genetic_algo.train_net.mutator import Mutator


if __name__ == '__main__':
    sample_size = 100
    train_datafile = 'train1.txt'
    test_datafile = 'test1.txt'
    
    if len(sys.argv) > 2 and sys.argv[1] and sys.argv[2]:
        train_datafile = sys.argv[1]
        test_datafile = sys.argv[2]

    output_dir_path = './wnet1.txt'
    train_dataset, test_dataset = Dataset(train_datafile, batch_size=256), Dataset(test_datafile, batch_size=256)
    layer_dims = [len(test_dataset[0].sample), 8, 4, 2, 1]
    activations = [Swish] * (len(layer_dims) - 1)
    args: SimulationArgs = SimulationArgs(fitness_goal=1, elite_percentile=0.99, mutation_percentage=0.8, 
                                          mutation_threshold=0.5, mutation_magnitude=0.05)
    simulator = Simulator(GeneticAlgorithmType.REGULAR, sample_size, train_dataset, test_dataset, output_dir_path, args, max_steps=500)
    simulator.run(layer_dims, activations, Mutator.mutate_additive)
