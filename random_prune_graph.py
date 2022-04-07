import os.path as osp
import argparse
import os
import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from pytorch_train import *
import numpy as np

# before wrong pipeline: pretrain -> prune -> test

# current right pipeline : prune -> retrain -> test

def prune_adj(oriadj:torch.Tensor, percent:int) -> torch.Tensor:
    adj = np.copy(oriadj.detach().cpu().numpy())
    low_adj = np.tril(adj, -1)
    i,j = np.nonzero(low_adj)
    ix = np.random.choice(len(i), int(np.floor(percent / 100 * len(i))), replace=False) # randomly choose ix
    low_adj[i[ix],j[ix]] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))
    return torch.from_numpy(adj).to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--ratio_graph', type=int, default=99)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--save_file', type=str, default="model.pth.tar")
parser.add_argument("--dataset", type=str, default="Cora")
args = parser.parse_args()
# get dataset
dataset = args.dataset
print("random_prune_graph dataset: ",dataset)
checkpoint = torch.load(f"./pretrain_pytorch/{dataset}_model.pth.tar")
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
model, data = Net(dataset, data, args).to(device), data.to(device)
model.load_state_dict(checkpoint)
# strat pruning
adj1,adj2 = model.adj1, model.adj2
id1 = torch.eye(adj1.shape[0]).to(device)
id2 = torch.eye(adj2.shape[0]).to(device)
adj1 = prune_adj(adj1 - id1, args.ratio_graph)
adj2 = prune_adj(adj2 - id2, args.ratio_graph)
model.adj1 = adj1
model.adj2 = adj2

# print("Randomly Prune Finished!")
# train_acc, val_acc, tmp_test_acc = test(model, data)
log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# print(log.format(args.ratio_graph, train_acc, val_acc, tmp_test_acc))
log_4_test = 'Tune Ratio: {:d}'
cur_adj1 = model.adj1.cpu().numpy()
cur_adj2 = model.adj2.cpu().numpy()
# print("finish L1 training, num of edges * 2 + diag in adj1:", np.count_nonzero(cur_adj1))
# print("finish L1 training, num of edges * 2 + diag in adj2:", np.count_nonzero(cur_adj2))

model_name = "./graph_random_prune/"+args.save_file
os.system("rm "+model_name)
torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./graph_random_prune/{args.save_file}")

# retrain
os.system("python3 "+"pytorch_retrain_with_graph.py"+" --load_path "+model_name+" --dataset "+str(args.dataset))


