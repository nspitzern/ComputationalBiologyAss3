from typing import List
from collections import namedtuple

from numpy import fromstring

DataItem = namedtuple('item', ['sample', 'label'])


def load_data_file(filepath: str) -> List[DataItem]:
    items = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        sample, label = line.strip().split()

        items.append(DataItem(fromstring(sample, 'u1') - ord('0'), int(label)))

    return items
