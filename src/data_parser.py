from typing import List, Tuple

from numpy import fromstring, ndarray


def load_data_file(filepath: str) -> Tuple[List[ndarray], List[int]]:
    samples, labels = [], []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        sample, label = line.strip().split()
        samples.append(fromstring(sample,'u1') - ord('0'))
        labels.append(int(label))

    return samples, labels
