import sys

import numpy as np
import scipy.sparse as sp
import torch
# import torch_sparse
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize, aug_normalized_adjacency
from time import perf_counter
import ipdb
from utils import *
import os.path as osp
from torch_sparse import coalesce, SparseTensor
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T

from sklearn import metrics

def get_balanced_random_split(one_hot_labels, num_classes, percls_trn=20, val_lb=500):
    indices = []
    labels = np.argmax(one_hot_labels, axis=1)
    for i in range(num_classes):
        index = np.nonzero(labels==i)[0]
        # print(f'i: {i}, index: {index}')
        index = np.random.permutation(index)
        indices.append(index)
    # print([i[:percls_trn] for i in indices])
    train_index = np.concatenate([i[:percls_trn] for i in indices])
    # print(f'train_index: {train_index}')
    rest_index = np.concatenate([i[percls_trn:] for i in indices])
    rest_index = np.random.permutation(rest_index)
    # print(f'rest_index: {rest_index}')
    val_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]
    return train_index, val_index, test_index


def load_citation_syn_chain_IDM(normalization, cuda, num_chains, chain_len, num_class=2,
                                noise=0.00, noise_type=None, need_orig=False):
    """load the synthetic dataset: chain"""
    # r = np.random.RandomState(42)
    c = num_class # num of classes
    n = num_chains # chains for each class
    l = chain_len # length of chain
    f = 100 # feature dimension
    # tn = 20  # train nodes
    # vl = 100 # val nodes
    # tt = 200 # test nodes
    num_nodes = c*n*l
    tn = int(num_nodes*0.05)
    vl = int(num_nodes*0.1)
    tt = num_nodes - tn - vl
    noise = noise

    # directed chains
    chain_adj = sp.coo_matrix((np.ones(l-1), (np.arange(l-1), np.arange(1, l))), shape=(l, l))
    adj = sp.block_diag([chain_adj for _ in range(c*n)]) # square matrix N = c*n*l

    # features = r.uniform(-noise, noise, size=(c, n, l, f))
    if noise_type is None:
        features = np.random.uniform(-noise, noise, size=(c,n,l,f))
    elif noise_type == 'normal':
        features = np.random.normal(0, noise, size=(c,n,l,f))
    elif noise_type == 'normal_v2':
        features = np.random.normal(0, 1, size=(c,n,l,f)) * noise
    else:
        raise RuntimeError(f'Cannot find noise type {noise_type}')
    #features = np.zeros_like(features)
    features[:, :, 0, :c] += np.eye(c).reshape(c, 1, c) # add class info to the first node of chains.
    features = features.reshape(-1, f)

    labels = np.eye(c).reshape(c, 1, 1, c).repeat(n, axis=1).repeat(l, axis=2) # one-hot labels
    labels = labels.reshape(-1, c)
    # ipdb.set_trace()

    idx_random = np.arange(c*n*l)
    # r.shuffle(idx_random)
    np.random.shuffle(idx_random)
    idx_train = idx_random[:tn]
    idx_val = idx_random[tn:tn+vl]
    idx_test = idx_random[tn+vl:tn+vl+tt]
    print(f'idx_train: {len(idx_train)}, idx_val: {len(idx_val)}, idx_test: {len(idx_test)}')
    # print(f'idx_train: {idx_train}')
    print(f'features[0,:10]: {features[0,:10]}')
    print(f'features[1,:10]: {features[1,:10]}')

    if need_orig:
        adj_orig = aug_normalized_adjacency(adj, need_orig=True)
        adj_orig = sparse_mx_to_torch_sparse_tensor(adj_orig).float()
        if cuda:
            adj_orig = adj_orig.cuda()

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj, features = preprocess_citation(adj, features, normalization)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense() if sp.issparse(features) else features)).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    sp_adj = adj
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, sp_adj, features, labels, idx_train, idx_val, idx_test


def load_citation_syn_chain(normalization, cuda, num_chains, chain_len, num_class=2, noise=0.00,
                            noise_type=None, need_orig=False):
    """load the synthetic dataset: chain"""
    c = num_class # num of classes
    n = num_chains # chains for each class
    l = chain_len # length of chain
    f = 100 # feature dimension
    num_nodes = c*n*l
    tn = int(num_nodes*0.05)
    vl = int(num_nodes*0.1)
    tt = num_nodes - tn - vl
    noise = noise

    chain_adj = sp.coo_matrix((np.ones(l-1), (np.arange(l-1), np.arange(1, l))), shape=(l, l))
    adj = sp.block_diag([chain_adj for _ in range(c*n)]) # square matrix N = c*n*l

    # features = r.uniform(-noise, noise, size=(c, n, l, f))
    # features = np.random.uniform(-noise, noise, size=(c,n,l,f))
    if noise_type is None:
        features = np.random.uniform(-noise, noise, size=(c,n,l,f))
    elif noise_type == 'normal':
        features = np.random.normal(0, noise, size=(c,n,l,f))
    elif noise_type == 'normal_v2':
        features = np.random.normal(0, 1, size=(c,n,l,f)) * noise
    else:
        raise RuntimeError(f'Cannot find noise type {noise_type}')
    #features = np.zeros_like(features)
    features[:, :, 0, :c] += np.eye(c).reshape(c, 1, c) # add class info to the first node of chains.
    features = features.reshape(-1, f)

    labels = np.eye(c).reshape(c, 1, 1, c).repeat(n, axis=1).repeat(l, axis=2) # one-hot labels
    labels = labels.reshape(-1, c)
    idx_random = np.arange(c*n*l)
    # r.shuffle(idx_random)
    np.random.shuffle(idx_random)
    idx_train = idx_random[:tn]
    idx_val = idx_random[tn:tn+vl]
    idx_test = idx_random[tn+vl:tn+vl+tt]
    print(f'idx_train: {len(idx_train)}, idx_val: {len(idx_val)}, idx_test: {len(idx_test)}')
    print(f'labels of idx_train: {labels[idx_train, :]}')
    # print(f'idx_train: {idx_train}')
    print(f'features[0,:10]: {features[0,:10]}')

    if need_orig:
        adj_orig = aug_normalized_adjacency(adj, need_orig=True)
        adj_orig = sparse_mx_to_torch_sparse_tensor(adj_orig).float()
        if cuda:
            adj_orig = adj_orig.cuda()

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense() if sp.issparse(features) else features)).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return [adj, adj_orig] if need_orig else adj, features, labels, idx_train, idx_val, idx_test


class WebKB_new(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'wisconsin']

        super(WebKB_new, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.float)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of the
    number of average monthly traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Chameleon"`,
            :obj:`"Squirrel"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']

        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def get_heterophilic_dataset_IDM(dataset_name, data_path, idx_split):
    dataset_name = dataset_name.lower()
    assert dataset_name in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
    if dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB_new(data_path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(data_path, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]
    train_mask, val_mask, test_mask = data.train_mask[:, idx_split], data.val_mask[:, idx_split], \
                                      data.test_mask[:, idx_split]
    edge_index, x, y = data.edge_index, data.x, data.y
    y = y.long()
    row, col = edge_index[0, :], edge_index[1, :]
    val = np.ones(len(row))
    adj = sp.coo_matrix((val, (row,col)), shape=(x.size(0), x.size(0)))
    sp_adj = aug_normalized_adjacency(adj)
    adj = sparse_mx_to_torch_sparse_tensor(sp_adj, device='cuda')
    return adj, sp_adj, x, y, train_mask, val_mask, test_mask