from torch_geometric.datasets import Planetoid,AttributedGraphDataset, WebKB,HeterophilousGraphDataset, WikipediaNetwork,LINKXDataset,WikiCS,Actor,Amazon,Coauthor
import torch
import numpy as np
import os
from torch_geometric.data import Data
from torch_sparse import SparseTensor

import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric.data.dataset")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

def load_Sq_Cha_filterred(name):
    data = np.load(os.path.join('data/', f'{name}_filtered.npz'))

    node_features = torch.tensor(data['node_features'], dtype=torch.float)
    labels = torch.tensor(data['node_labels'], dtype=torch.long)
    num_nodes=len(labels)
    edges = torch.tensor(data['edges'], dtype=torch.long)
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])


    edge_index = edges.t().contiguous()
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

    data = Data(x=node_features, edge_index=edge_index,adj_t=adj_t, y=labels,train_mask=train_masks,val_mask=val_masks,test_mask=test_masks)

    return data

def load_data(dataset_Name,Trans):
    if dataset_Name=='cora':
        data_loaded = Planetoid(root='/tmp/cora', name='cora',transform=Trans)
    elif dataset_Name=='citeseer':
        data_loaded = Planetoid(root='/tmp/citeseer', name='citeseer',transform=Trans)
    elif dataset_Name=='pubmed':
        data_loaded = Planetoid(root='/tmp/pubmed', name='pubmed',transform=Trans)
    elif dataset_Name=='texas':
        data_loaded = WebKB(root='/tmp/texas', name='texas',transform=Trans)
    elif dataset_Name=='cornell':
        data_loaded = WebKB(root='/tmp/cornell', name='cornell',transform=Trans)
    elif dataset_Name=='wisconsin':
        data_loaded = WebKB(root='/tmp/wisconsin', name='wisconsin',transform=Trans)
    # elif dataset_Name=='chameleon':
    #     data_loaded = WikipediaNetwork(root='/tmp/chameleon', name='chameleon',transform=Trans)
    elif dataset_Name=='penn94':
        data_loaded = LINKXDataset(root='/tmp/Penn94', name='Penn94',transform=Trans)
    elif dataset_Name=='actor':
        data_loaded = Actor(root='/tmp/actor',transform=Trans)
    # elif dataset_Name=='squirrel':
    #     data_loaded = WikipediaNetwork(root='/tmp/squirrel', name='squirrel',transform=Trans)
    elif dataset_Name=='computers':
        data_loaded = Amazon(root='/tmp/Computers', name='Computers',transform=Trans)
    elif dataset_Name=='photo':
        data_loaded = Amazon(root='/tmp/Photo', name='Photo',transform=Trans)
    elif dataset_Name=='physics':
        data_loaded = Coauthor(root='/tmp/Physics', name='Physics',transform=Trans)
    elif dataset_Name=='cs':
        data_loaded = Coauthor(root='/tmp/CS', name='CS',transform=Trans)
    elif dataset_Name=='wikics':
        data_loaded =WikiCS(root='/tmp/wikics',transform=Trans)
    elif dataset_Name=='roman-empire':
        data_loaded =HeterophilousGraphDataset(root='/tmp/Roman-empire', name='Roman-empire',transform=Trans)
    elif dataset_Name=='amazon-ratings':
        data_loaded =HeterophilousGraphDataset(root='/tmp/Amazon-ratings', name='Amazon-ratings',transform=Trans)
    elif dataset_Name=='minesweeper':
        data_loaded =HeterophilousGraphDataset(root='/tmp/Minesweeper', name='Minesweeper',transform=Trans)
    elif dataset_Name=='questions':
        data_loaded =HeterophilousGraphDataset(root='/tmp/Questions', name='Questions',transform=Trans)
    elif dataset_Name=='tolokers':
        data_loaded =HeterophilousGraphDataset(root='/tmp/tolokers', name='tolokers',transform=Trans)
    elif dataset_Name=='flickr':
        data_loaded = AttributedGraphDataset(root='/tmp/Flickr', name='Flickr',transform=Trans)
    elif dataset_Name=='blogcatalog':
        data_loaded = AttributedGraphDataset(root='/tmp/BlogCatalog', name='BlogCatalog',transform=Trans)
    else:
        raise NotImplementedError
    return data_loaded