import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from pytorch_train import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument("--draw", type=int, default=100)
parser.add_argument('--ratio_graph', type=int, default=90)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--save_file', type=str, default="model.pth.tar")
parser.add_argument("--dataset", type=str, default="CiteSeer")
args = parser.parse_args()

dataset = args.dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0]
model, data = Net(dataset, data, args).to(device), data.to(device)
checkpoint = torch.load(f"./pretrain_pytorch/{args.dataset}_model.pth.tar")
model.load_state_dict(checkpoint)

loss = lambda m: F.nll_loss(m()[data.train_mask], data.y[data.train_mask])
# print("construct admm training")
support1 = model.adj1
support2 = model.adj2
partial_adj_mask = support1.cpu().numpy()
# print("num of edges * 2 + diag in adj:", np.count_nonzero(partial_adj_mask))
adj_variables = [support1,support2]
rho = 1e-3
non_zero_idx = np.count_nonzero(support1.cpu().numpy())
Z1 = U1 = Z2 = U2 = torch.from_numpy(np.zeros_like(partial_adj_mask)).to(device)
model.adj1.requires_grad = True
model.adj2.requires_grad = True

# Update the gradient of the adjacency matrices
# grads_vars: {name: torch.Tensor}
def update_gradients_adj(grads_vars ,adj_p_mask:np.ndarray):
    temp_grad_adj1 = 0
    var1 = None
    var2 = None
    temp_grad_adj2 = 0
    for key,var in grads_vars.items():
        grad = var.grad
        print(grad[grad != 0])
        if key == "support1":
            adj_mask = torch.from_numpy(adj_p_mask).to(device)
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
            adj_mask = torch.from_numpy(adj_p_mask).to(device)
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
            var2 = var
    grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 # Why are we doing this?
    var1.grad = grad_adj
    var2.grad = grad_adj
    return [var1,var2]

def prune_adj(oriadj:torch.Tensor, non_zero_idx:int, percent:int) -> torch.Tensor:
    original_prune_num = int(((non_zero_idx - oriadj.shape[0]) / 2) * (percent / 100))
    adj = np.copy(oriadj.detach().cpu().numpy())
    # print(f"Pruning {percent}%")
    low_adj = np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    # print(adj.shape[0],original_prune_num,before,after, before-after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))
    return torch.from_numpy(adj).to(device)

# Define new loss function
admm_loss = lambda m: loss(m) + \
            rho * (F.mse_loss(support1 + U1, Z1 + torch.eye(support1.shape[0]).to(device)) +
            F.mse_loss(support2 + U2, Z2 + torch.eye(support2.shape[0]).to(device)))
adj_optimizer = torch.optim.Adam(adj_variables,lr=0.001)
adj_map = {"support1": support1, "support2": support2}
id1 = torch.eye(support1.shape[0]).to(device)
id2 = torch.eye(support2.shape[0]).to(device)

cnt = 0
best_prune_acc = 0
for j in range(args.times):
    for epoch in range(args.epochs):
        if cnt == args.draw:
            break
        model.train()
        adj_optimizer.zero_grad()
        # Calculate gradient
        admm_loss(model).backward(retain_graph=True)
        # Update to correct gradient
        update_gradients_adj(adj_map, partial_adj_mask)
        # Use the optimizer to update adjacency matrix
        adj_optimizer.step()

        train_acc, val_acc, tmp_test_acc = test(model, data)
        if val_acc > best_prune_acc:
            best_prune_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Pruning Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        cnt += 1
        # print(log.format(epoch, train_acc, best_prune_acc, test_acc))

    # Use learnt U1, Z1 and so on to prune
    adj1,adj2 = model.adj1, model.adj2
    Z1 = adj1 - id1 + U1
    Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
    U1 = U1 + (adj1 - id1 - Z1)

    Z2 = adj2 - id2 + U2
    Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
    U2 = U2 + (adj2 - id2 - Z2)
    
    if cnt == args.draw:
        break    

adj1,adj2 = model.adj1, model.adj2
adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
model.adj1 = adj1
model.adj2 = adj2

# print("Optimization Finished!")
train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
log_4_test = 'Tune Ratio: {:d}'
# print(log_4_test.format(args.ratio))
cur_adj1 = model.adj1.cpu().numpy()
cur_adj2 = model.adj2.cpu().numpy()
print("finish L1 training, num of edges * 2 + diag in adj1:", np.count_nonzero(cur_adj1))
# print("finish L1 training, num of edges * 2 + diag in adj2:", np.count_nonzero(cur_adj2))
# print("symmetry result adj1: ", testsymmetry(cur_adj1))
# print("symmetry result adj2: ", testsymmetry(cur_adj2))
# print("is equal of two adj", isequal(cur_adj1, cur_adj2))

torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./graph_pruned_pytorch/{args.save_file}")





