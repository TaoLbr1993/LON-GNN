import os
import os.path as osp
import pickle

import torch
import torch.nn as nn
import torch_sparse
import torch_geometric as thg
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import to_undirected

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):
        if name=='actor':
            name='film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

class WebKB(InMemoryDataset):
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
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']
        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

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
            edge_index = to_undirected(edge_index)
            edge_index, _ = torch_sparse.coalesce(edge_index, None, x.size(0), x.size(0))
        adj_t = torch_sparse.SparseTensor.from_edge_index(edge_index)
        data = Data(x=x, edge_index=edge_index, y=y, adj_t=adj_t)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)

def load_dataset(ds_name, ds_fp=None):
    '''
        Part of this function is adapted from https://github.com/jianhao2016/GPRGNN/blob/master/src/dataset_utils.py
    '''
    if ds_fp is None:
        dsfp_name = 'data'
        root_dir = os.path.dirname(os.path.abspath(__file__))
        ds_fp = os.path.join(root_dir, dsfp_name)
    else:
        dsfp_name = ds_fp
    if ds_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(ds_fp, ds_name, transform=NormalizeFeatures())
    
    elif ds_name in ['cornell', 'texas', 'wisconsin']:
        def pre_tran(data):
            edge_index = data.edge_index
            edge_index = thg.utils.to_undirected(edge_index)
            edge_index, _ = torch_sparse.coalesce(edge_index, None, data.y.shape[0], data.y.shape[0])
            data.edge_index = edge_index
            adj_t = torch_sparse.SparseTensor.from_edge_index(edge_index)
            data.adj_t = adj_t
            return data
        dataset = WebKB(root=ds_fp, name=ds_name, transform=NormalizeFeatures())

    elif ds_name in ['computers', 'photo']:
        dataset = Amazon(ds_fp, ds_name, transform=NormalizeFeatures())

    elif ds_name in ['chameleon', 'squirrel', 'actor']:
        dataset = dataset_heterophily(root='./data/', name=ds_name, transform=NormalizeFeatures())

    else:
        assert 1==0, "invalid dataset"

    return dataset


if __name__ == '__main__':
    dataset = load_dataset("cornell")
    print(dataset)
    data = dataset[0]
    print(data.x)