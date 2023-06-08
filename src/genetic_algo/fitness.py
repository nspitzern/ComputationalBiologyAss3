import numpy as np

from src.dataset import Dataset
from src.genetic_algo.sample import Sample

def correctness_ratio(sample: Sample, dataset: Dataset) -> float:
    """
    Check the ratio of decoded words that appear in the corpus
    :param dec: List of decoded words
    :param corpus: List of Corpus words
    :return: float: ratio
    """
    return np.mean([1 if sample(s) == label else 0 for s, label in dataset])
