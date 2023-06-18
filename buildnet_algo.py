from src.genetic_algo.build_net.simulator import SimulationArgs, Simulator
from src.dataset import Dataset
from src.genetic_algo.build_net.strategy import GeneticAlgorithmType
from src.genetic_algo.train_net.mutator import Mutator


if __name__ == '__main__':
    sample_size = 100
    datafile = 'nn0.txt'
    output_dir_path = f'output/buildnet_algo/{datafile.split(".")[0]}'
    dataset: Dataset = Dataset(datafile, batch_size=256)
    net_optional_sizes = [2 ** x for x in range(4)]
    args: SimulationArgs = SimulationArgs(fitness_goal=1, elite_percentile=0.99, mutation_function=Mutator.mutate_additive,
                                          mutation_percentage=0.8, mutation_threshold=0.3, mutation_magnitude=0.05)
    simulator = Simulator(GeneticAlgorithmType.REGULAR, sample_size, dataset, output_dir_path, args, max_steps=3)
    simulator.run(net_optional_sizes)
