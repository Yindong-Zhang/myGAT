from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from collections import deque
from dataset.make_dataset import get_dataset
import numpy as np
import random
def bfs(start, adj, distance):
    """

    :param start:
    :param adj: a sparse adjacent matrix
    :param distance:
    :return:
    """
    num_nodes = adj.shape[0]
    visited = [False, ] * num_nodes
    q = deque()
    q.append((start, 0))
    visited[start] = True
    node_list = [start, ]
    while(len(q) != 0):
        cur_node, cur_dist = q.pop()
        node_list.append(cur_node)
        if(cur_dist + 1 > distance):
            break
        for next_node in adj[cur_node].nonzero()[1]:
            if not visited[next_node]:
                q.append((next_node, cur_dist + 1))
                visited[next_node] = True

    while(len(q) != 0):
        node, dist = q.pop()
        node_list.append(node)

    return node_list

def bfs_sample(start, adj, distance, sample_num):
    """

    :param start:
    :param adj:
    :param distance:
    :param sample_numbers: should be a list specific number of support node sampled at each distance
    :return:
    """
    assert distance == len(sample_num), "BFS distance should equal to length of sample_nums."
    num_nodes = adj.shape[0]
    visited = [False, ] * num_nodes
    nodelist = [start, ]
    curlist = [start, ]
    visited[start] = True
    for i in range(distance):
        nextlist = []
        for cur_node in curlist:
            downstream = []
            next_nodes = adj[cur_node].nonzero()[1]
            for node in next_nodes:
                if not visited[node]:
                    downstream.append(node)
            if len(downstream) > sample_num[i]:
                random.shuffle(downstream)
                downstream = downstream[:sample_num[i]]

            for node in downstream:
                visited[node] = True
            nextlist.extend(downstream)



        nodelist.extend(nextlist)
        curlist = nextlist

    return nodelist



class SubGraph(Dataset):
    def __init__(self, adj, features, labels, idx, num_layers, num_samples):
        """

        :param adj: suppose adj a sparse adjacent matrix
        :param features:
        :param labels:
        :param num_layers:
        """
        self.adj = adj
        self.features = features
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.labels = labels
        self.idx = idx
        self.num_nodes = len(idx)

    def __getitem__(self, item):
        node = self.idx[item]
        nodelist = bfs_sample(node, self.adj, self.num_layers, self.num_samples)
        min_adj = self.adj[nodelist][:, nodelist]
        min_features = self.features[nodelist]
        min_label = self.labels[node]
        return (min_adj, min_features), min_label

    def __len__(self):
        return self.num_nodes

def custom_collate(batch):
    """

    :param batch: a list of tuple ((adj, features), label)
    :return:
    """
    max_nodes = max(map(lambda x: x[0][0].shape[0], batch))
    batch = [align(data, max_nodes) for data in batch]
    return default_collate(batch)

def align(data_tuple, max_nodes):
    """
    consider adj as sparse matrix and features as dense ndarray.
    :param data_tuple:
    :param max_nodes:
    :return:
    """
    (adj, features), label = data_tuple
    adj.resize(max_nodes, max_nodes)
    adj_fill = adj.toarray()
    features_fill = np.zeros((max_nodes, features.shape[1]), )
    features_fill[:features.shape[0]] = features
    return (adj_fill, features_fill), label

if __name__ == "__main__":
    # data_name = "citeseer"
    # adj, features, labels = get_dataset(data_name, './data/npz/{}.npz'.format(data_name), standardize=True,
    #                                     train_examples_per_class=40, val_examples_per_class=100)
    # features = features.toarray()
    from utils import load_reddit_in_networkx
    import networkx as nx
    G, features, id_map, walks, class_map = load_reddit_in_networkx()
    nodelist = sorted(id_map, key= lambda x: id_map[x])
    adj = nx.adjacency_matrix(G, nodelist)
    labels = [class_map[node] for node in nodelist]
    # %%
    subgraph = SubGraph(adj, features, labels, np.arange(adj.shape[0]), 3, [25, 10, 5])
    dataloader = DataLoader(subgraph, batch_size = 2, num_workers= 4, collate_fn= custom_collate)
    # %%
    for data in dataloader:
        (min_adj, min_feat), label = data
        print(min_adj.shape, min_feat.shape, label)