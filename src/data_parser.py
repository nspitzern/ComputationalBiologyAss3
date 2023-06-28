from typing import List
from collections import namedtuple

from numpy import fromstring

DataItem = namedtuple('item', ['sample', 'label'])


def load_data_file(filepath: str, test: bool = False) -> List[DataItem]:
    items = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if not test:
            sample, label = line.strip().split()
        else:
            sample = line.strip()

        if not test:
            items.append(DataItem(fromstring(sample, 'u1') - ord('0'), int(label)))
        else:
            items.append(fromstring(sample, 'u1') - ord('0'))

    return items
