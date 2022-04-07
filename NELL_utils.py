import sys
import os.path as osp
from itertools import repeat

import torch
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_planetoid_data(folder, prefix):
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    test_idx_range  = test_index.sort()[0]
    
    if prefix.startswith('nell'):
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                            dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended, dtype=np.float32)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        idx_train = np.arange(x.shape[0])
        idx_test = test_idx_reorder
        if model_config['validate']:
            assert x.shape[0] + validation_size < allx.shape[0] + tx.shape[0]
            idx_val = np.arange(x.shape[0], x.shape[0] + validation_size)
        else:
            idx_val = test_idx_reorder

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    x[test_index] = x[sorted_test_index]
    y[test_index] = y[sorted_test_index]


    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    out = torch.Tensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
    # NOTE: There are duplicated edges and self loops in the datasets. Other
    # implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
