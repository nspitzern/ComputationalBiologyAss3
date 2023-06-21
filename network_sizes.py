from src.network import Network, load_network


if __name__ == '__main__':
    filename = 'wnet1.json'
    network: Network = load_network(filename)

    layer_dims = []
    layer_dims.append(network.layers[0].shape[0])
    layer_dims.append(network.layers[0].shape[1])
    layer_dims.extend([l.shape[1] for l in network.layers[1:]])
    print(layer_dims)