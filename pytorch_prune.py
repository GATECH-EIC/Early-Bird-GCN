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
parser.add_argument('--ratio', type=int, default=90)
parser.add_argument("--load_file", type=str, default="model.pth.tar")
parser.add_argument("--save_file", type=str, default="model.pth.tar")
parser.add_argument('--use_gdc', type=bool, default=False)
args = parser.parse_args()

dataset = 'CiteSeer'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# print(f"Number of graphs in {dataset} dataset:", len(dataset))
data = dataset[0]

checkpoint = torch.load(f"./graph_pruned_pytorch/{args.load_file}")
adj = checkpoint["adj"]
state_dict = checkpoint["state_dict"]
model, data = Net(dataset, data, args,adj=adj).to(device), data.to(device)
model.load_state_dict(state_dict)

train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'Loaded model with accuracy: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(train_acc, val_acc, tmp_test_acc))

############################ global weight pruning #############################
total = 0
for m in model.modules():
    if isinstance(m, GCNConv):
        total += m.weight.data.numel()
conv_weights = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, GCNConv):
        size = m.weight.data.numel()
        conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
        index += size
y, i = torch.sort(conv_weights)
thre_index = int(total * args.ratio / 100)
thre = y[thre_index]
pruned = 0
# print('Pruning threshold: {}'.format(thre))
zero_flag = False
for k, m in enumerate(model.modules()):
    if isinstance(m, GCNConv):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().to(device)
        pruned = pruned + mask.numel() - torch.sum(mask)
        m.weight.data.mul_(mask)
        if int(torch.sum(mask)) == 0:
            zero_flag = True
        # print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
        #       format(k, mask.numel(), int(torch.sum(mask))))
# print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))

#######################
# print("\nTesting")

train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After prune: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
log_4_test = 'After prune: Ratio: {:d}, Test: {:.4f}'
# print(log_4_test.format(args.ratio, tmp_test_acc))

torch.save({"state_dict":model.state_dict(),"adj":adj}, f"./pruned_pytorch/{args.save_file}")


