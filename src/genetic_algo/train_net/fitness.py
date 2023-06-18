import numpy as np

from src.dataset import Dataset
from src.genetic_algo.train_net.sample import Sample

def correctness_ratio(sample: Sample, dataset: Dataset) -> float:
    """
    Check the ratio of correct predictions
    Args:
        sample (Sample): the sample to check
        dataset (Dataset): the dataset to test the sample on

    Returns:
        float: ratio
    """
    return np.mean([sample(batch) == labels for batch, labels in dataset])
