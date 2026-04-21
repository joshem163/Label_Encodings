from dataloader import *  # Custom data loader script to import datasets
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split
from logger import *  # Custom logger to track model performance across runs
#from model import MPNNs,objectview
from models import *

from torch_geometric.nn import LINKX
import torch_geometric.transforms as T  # For transforming the graph data
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from torch_sparse import SparseTensor
from data_utils import load_fixed_splits
#from modules_lss import *
from module_gpu import *
from tqdm import tqdm
import time
from modules_lss import save_best_result

def is_binary_feature_matrix(x: torch.Tensor) -> bool:
    # works for float/int tensors
    unique_vals = torch.unique(x)
    return torch.all((unique_vals == 0) | (unique_vals == 1)).item()
def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    # Forward pass with GraphSAGE and MLP inputs
    out = model(data.x, data.adj_t)[train_idx]

    # Compute loss (using Cross Entropy instead of NLL since we don't apply log_softmax)
    loss = F.cross_entropy(out, data.y.squeeze()[train_idx])

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def ACC(prediction, label):
    correct = prediction.eq(label).sum().item()
    total = len(label)
    return correct / total


@torch.no_grad()
def test(model, data, train_idx, valid_idx, test_idx, metric='accuracy'):
    model.eval()
    out = model(data.x, data.adj_t)  # raw logits
    y_true = data.y.squeeze()  # ensure shape consistency

    if metric == 'accuracy':
        y_pred = out.argmax(dim=-1)
        train_score = ACC(y_pred[train_idx], y_true[train_idx])
        valid_score = ACC(y_pred[valid_idx], y_true[valid_idx])
        test_score = ACC(y_pred[test_idx], y_true[test_idx])

    elif metric == 'roc_auc':
        # Assume binary classification and get probability of class 1
        probs = F.softmax(out, dim=-1)[:, 1]  # get prob for class 1

        train_score = roc_auc_score(y_true[train_idx].cpu(), probs[train_idx].cpu())
        valid_score = roc_auc_score(y_true[valid_idx].cpu(), probs[valid_idx].cpu())
        test_score = roc_auc_score(y_true[test_idx].cpu(), probs[test_idx].cpu())

    else:
        raise ValueError("Unsupported metric: choose 'accuracy' or 'roc_auc'")

    return train_score, valid_score, test_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_mask_indices(mask, run=None):
    mask = mask.cpu()  # Ensure the mask tensor is on CPU
    if run is None:
        return np.where(mask.numpy())[0]
    else:
        return np.where([mask[i][run].item() for i in range(len(mask))])[0]


def main(args):
    # Convert args to an object-like structure for easier access
    if isinstance(args, dict):
        args = objectview(args)
    # Logger to store and output results
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        if args.dataset in ['chameleon', 'squirrel']:
            data = load_Sq_Cha_filterred(args.dataset)
            split_idx_lst = load_fixed_splits('data', args.dataset, name=args.dataset)
            out_dim = 5
        else:
            dataset = load_data(args.dataset, None)
            data = dataset[0]  # First graph data object
            out_dim = dataset.num_classes
        # # Split data into training, validation, and test based on public split or custom split

            # Generic fallback with 60/20/20 split
        num_nodes = data.x.size(0)
        print('number of node ',num_nodes)
        all_indices = np.arange(num_nodes)

        train_idx, temp_idx = train_test_split(
            all_indices,
            train_size=0.6,
            random_state=run + 123,
            shuffle=True
        )

        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=run + 123,
            shuffle=True
        )

        # ----- Proto-Embeddings -----
        if isinstance(data.x, torch.Tensor) and data.x.layout != torch.strided:
            data.x = data.x.to_dense().float()
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        valid_idx = torch.tensor(valid_idx, dtype=torch.long)
        test_idx = torch.tensor(test_idx, dtype=torch.long)

        num_nodes = data.edge_index.max().item() + 1  # total number of nodes

        # Convert edge_index to SparseTensor (adj_t)
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))
        # print(data)
        #model=MPNNs(data.num_features, args.hidden_channels,out_dim)

        # # Reset parameters of the models to ensure fresh training
        if args.model_type == 'GCN':
            model = GCN(data.num_features, args.hidden_channels,out_dim , args.num_layers, args.dropout)
        elif args.model_type == 'SAGE':
            model = SAGE(data.num_features, args.hidden_channels, out_dim, args.num_layers, args.dropout)
        elif args.model_type == 'GAT':
            model = GAT(data.num_features, args.hidden_channels, out_dim, args.num_layers, args.dropout)
        elif args.model_type == 'LINKX':
            model = LINKX(len(data.y), data.num_features, args.hidden_channels, out_dim, args.num_layers, 1,
                          1,
                          args.dropout)
        else:
            print('Model does not exist')

        # Optimizers for each model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=5e-4)

        # Training loop for each epoch
        #for epoch in tqdm(range(1, 1 + args.epochs)):
        start_time = time.perf_counter()
        for epoch in range(1, 1 + args.epochs):
            # Train the model on the current run's data
            loss = train(model,data, train_idx, optimizer)
            # Test the model on validation and test sets
            if args.dataset in ['questions', 'minesweeper','tolokers']:
                result = test(model, data, train_idx, valid_idx, test_idx, metric='roc_auc')
            else:
                result = test(model, data, train_idx, valid_idx, test_idx, metric='accuracy')
            logger.add_result(run, result)  # Log the results

            # Log results every `log_steps` epochs
            #if epoch % args.log_steps == 0:
             #   train_acc, valid_acc, test_acc = result
              #  print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, '
               #       f'Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')

        # Print run statistics
        #logger.print_statistics(run)
        train_time = time.perf_counter() - start_time
        print(f"training time: {train_time:.4f} seconds")

    # Print overall statistics after all runs
    # logger.print_statistics()
    best_test = logger.print_statistics()  # <<< logger.print_statistics returns best test acc
    return best_test[0],best_test[1]  # <<< RETURN it!

import itertools
if __name__ == "__main__":
    total_start_time = time.time()

    #datasets_to_run =  ['cora','citeseer','pubmed','physics','cs','computers','photo','chameleon', 'squirrel','actor','amazon-ratings','questions','tolokers','wikics','blogcatalog','flickr','minesweeper']
    datasets_to_run=['cora','citeseer','texas','cornell','wisconsin']
    lr_values = [0.001]
    hidden_channels_values = [64]
    dropout_values = [0.5]
    layer=[2]

    for ds in datasets_to_run:
        print(f"\nRunning grid search on dataset: {ds}")

        best_result = -1
        std = 0
        best_args = None
        best_runtime = 0

        for lr, hidden_channels, dropout,layer_num in itertools.product(
            lr_values, hidden_channels_values, dropout_values,layer
        ):
            args = {
                'model_type': 'SAGE',
                'dataset': ds,
                'num_layers': layer_num,
                'heads': 1,
                'batch_size': 32,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'epochs': 200,
                'opt': 'adam',
                'opt_scheduler': 'none',
                'opt_restart': 0,
                'runs': 5,
                'log_steps': 10,
                'weight_decay': 5e-4,
                'lr': lr,
            }

            print(f"Trying config: lr={lr}, hidden_channels={hidden_channels}, dropout={dropout}")

            start_time = time.time()
            test_acc, test_std = main(args)
            end_time = time.time()
            total_time = end_time - start_time

            print(f"Total runtime: {total_time:.2f} seconds")

            if test_acc > best_result:
                best_result = test_acc
                std = test_std
                best_args = args.copy()
                best_runtime = total_time

        print("\n========== Best result for dataset:", ds, "==========")
        print(f"Best test accuracy: {best_result:.4f} ± {std:.4f}")
        print(f"Best hyperparameters: {best_args}")

        save_best_result(ds, best_result, std, best_args, best_runtime)

    #print(f"\nAll results saved to {RESULTS_FILE}")