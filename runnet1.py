from src.dataset import Dataset
from src.network import load_network

if __name__ == '__main__':
    dataset = Dataset('nn1.txt')
    net = load_network('wnet1.json')

    count = 0
    preds = []
    for s, real in dataset:
        pred = net(s)

        for p, r in zip(pred, real):
            if p == r:
                count += 1

        preds.append(str(pred))

    print(f'acc: {count / len(dataset)}')
    with open('predictions1', 'w') as f:
        f.write('\n'.join(preds))