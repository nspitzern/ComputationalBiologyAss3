from src.dataset import Dataset
from src.network import load_network

if __name__ == '__main__':
    dataset = Dataset('nn0.txt')

    sample_size = len(dataset[0].sample)

    net = load_network('wnet0')

    preds = []
    for s, _ in dataset:
        pred = str(net(s))

        preds.append(pred)

    with open('predictions0', 'w') as f:
        f.write('\n'.join(preds))