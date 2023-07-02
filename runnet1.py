from src.data_parser import load_data_file
from src.network import load_network

if __name__ == '__main__':
    try:
        dataset = load_data_file('testnet1', test=True)
    except FileNotFoundError:
        dataset = load_data_file('testnet1.txt', test=True)

    net = load_network('wnet1.txt')

    count = 0
    preds = []
    # for s, real in dataset:
    #     pred = net(s)
    #
    #     for p, r in zip(pred, real):
    #         if p == r:
    #             count += 1
    #
    #     preds.append(str(pred))

    for sample in dataset:
        pred = net(sample)
        preds.append(str(*pred))

    # print(f'acc: {count / len(dataset)}')
    with open('predictions1', 'w') as f:
        f.write('\n'.join(preds))
