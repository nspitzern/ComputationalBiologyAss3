from typing import List, Tuple
from collections import namedtuple

from numpy import fromstring, ndarray

DataItem = namedtuple('item', ['sample', 'label'])


def load_data_file(filepath: str) -> List[DataItem]:
    items = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        sample, label = line.strip().split()
        # samples.append(fromstring(sample,'u1') - ord('0'))
        # labels.append(int(label))

        items.append(DataItem(fromstring(sample, 'u1') - ord('0'), int(label)))

    return items
