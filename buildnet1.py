import sys
from src.genetic_algo.build_net.simulator import SimulationArgs, Simulator
from src.dataset import Dataset
from src.genetic_algo.build_net.strategy import GeneticAlgorithmType
from src.genetic_algo.train_net.mutator import Mutator


if __name__ == '__main__':
    sample_size = 10
    inner_samples = 50
    train_datafile = 'train1.txt'
    test_datafile = 'test1.txt'
    
    if len(sys.argv) > 2 and sys.argv[1] and sys.argv[2]:
        train_datafile = sys.argv[1]
        test_datafile = sys.argv[2]

    output_dir_path = './wnet1.txt'
    train_dataset, test_dataset = Dataset(train_datafile, batch_size=256), Dataset(test_datafile, batch_size=256)
    net_optional_sizes = [2 * x for x in range(1, 9)]
    args: SimulationArgs = SimulationArgs(fitness_goal=1, elite_percentile=0.99, mutation_function=Mutator.mutate_additive,
                                          mutation_percentage=0.25, mutation_threshold=0.3, mutation_magnitude=0.05)
    simulator = Simulator(GeneticAlgorithmType.REGULAR, sample_size, inner_samples, train_dataset, test_dataset, output_dir_path, 
                          args, max_steps=300)
    simulator.run(net_optional_sizes)
