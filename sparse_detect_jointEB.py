import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.utils.num_nodes as geo_num_nodes
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noga
from utils import *
from sparse_pytorch_train import *
import numpy as np
import logging
import os
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix, tril
from scipy import sparse
import torch.sparse as ts

# Update the gradient of the adjacency matrices
# grads_vars: {name: torch.Tensor}
def update_gradients_adj(grads_vars, adj_mask):
    temp_grad_adj1 = 0
    var1 = None
    var2 = None
    temp_grad_adj2 = 0
    for key,var in grads_vars.items():
        grad = var.grad
        if key == "support1":
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = temp_grad_adj.t_()
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
            temp_grad_adj = adj_mask * grad
            transposed_temp_grad_adj = temp_grad_adj.t_()
            temp_grad_adj2 = temp_grad_adj + transposed_temp_grad_adj
            var2 = var
    grad_adj = (temp_grad_adj1 + temp_grad_adj2) / 4 # Why are we doing this?
    var1.grad = grad_adj
    var2.grad = grad_adj
    return [var1,var2]

def prune_adj(oriadj, non_zero_idx:int, percent:int):
    original_prune_num = int(((non_zero_idx - oriadj.size()[0]) / 2) * (percent / 100))
    adj = SparseTensor.from_torch_sparse_coo_tensor(oriadj).to_scipy()

    # find the lower half of the matrix
    low_adj = tril(adj, -1)
    non_zero_low_adj = low_adj.data[low_adj.data != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj.data) < low_pcen
    before = len(non_zero_low_adj)
    low_adj.data[under_threshold] = 0
    non_zero_low_adj = low_adj.data[low_adj.data != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj.data != 0)
        low_adj.data[low_adj.data == 0] = 2000000
        flat_indices = np.argpartition(low_adj.data, rest_pruned - 1)[:rest_pruned]
        low_adj.data = np.multiply(low_adj.data, mask_low_adj)
        low_adj.data[flat_indices] = 0
    low_adj.eliminate_zeros()
    new_adj = low_adj + low_adj.transpose()
    new_adj = new_adj + sparse.eye(new_adj.shape[0])
    return SparseTensor.from_scipy(new_adj).to_torch_sparse_coo_tensor().to(device)

# torch.sparse
def get_mask(oriadj, non_zero_idx:int, percent:int):
    original_prune_num = int(((non_zero_idx - oriadj.size()[0]) / 2) * (percent / 100))
    adj = SparseTensor.from_torch_sparse_coo_tensor(oriadj).to_scipy()

    # find the lower half of the matrix
    low_adj = tril(adj, -1)
    non_zero_low_adj = low_adj.data[low_adj.data != 0]

    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    under_threshold = abs(low_adj.data) < low_pcen
    before = len(non_zero_low_adj)
    low_adj.data[under_threshold] = 0
    non_zero_low_adj = low_adj.data[low_adj.data != 0]
    after = len(non_zero_low_adj)

    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj.data != 0)
        low_adj.data[low_adj.data == 0] = 2000000
        flat_indices = np.argpartition(low_adj.data, rest_pruned - 1)[:rest_pruned]
        low_adj.data = np.multiply(low_adj.data, mask_low_adj)
        low_adj.data[flat_indices] = 0
    low_adj.eliminate_zeros()
    new_adj = low_adj + low_adj.transpose()
    new_adj = new_adj + sparse.eye(new_adj.shape[0])
    return SparseTensor.from_scipy((new_adj != adj)).to_torch_sparse_coo_tensor().int()

def calc_dist(m1,m2):
    diff = m1 - m2
    neg = diff < 0
    diff[neg] = -diff[neg]
    return torch.sum(diff.coalesce().values())

def post_processing():
    adj1,adj2 = model.adj1, model.adj2
    adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
    adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
    model.adj1 = adj1.float()
    model.adj2 = adj2.float()

    # print("Optimization Finished!")
    # train_acc, val_acc, tmp_test_acc = test(model, data)
    log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
    log_4_test = 'Tune Ratio: {:d}'
    # print(log_4_test.format(args.ratio))
    # cur_adj1 = model.adj1.cpu().numpy()
    # cur_adj2 = model.adj2.cpu().numpy()

    torch.save({"state_dict":model.state_dict(),"adj":model.adj1}, f"./graph_pruned_eb_pytorch/{args.save_file}")


parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--ratio_graph', type=int, default=90)
parser.add_argument('--ratio_weight', type=int, default=90)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--save_file', type=str, default="model.pth.tar")
parser.add_argument('--lookback', type=int, default=3)
parser.add_argument("--thres", type=float, default=0.0)
parser.add_argument("--dataset", type=str, default="CiteSeer")
parser.add_argument("--log", type=str, default="{:05d}")
args = parser.parse_args()

models = ["pruned_pytorch/model.pth.tar","prune_weight_cotrain/model.pth.tar","prune_weight_iterate/model.pth.tar","prune_weight_first/model.pth.tar"]
txts_wc = "test_weight_changes.txt"
res_list = []
os.system("rm "+txts_wc)
g_r_list = [20,40,60,80]
w_r_list = [50,70,90]
os.system("rm "+"./pretrain_pytorch/model.pth.tar")
os.system("CUDA_VISIBLE_DEVICES=0 python3 "+"pytorch_train.py"+" --epochs "+str(1)) # test without pretrain, prune from scratch

for i in range(4): # for each graph ratio [20,40,60,80]
    g_ratio = g_r_list[i]
    for j in range(3): # for each weight ratio [50,70,90]
        w_ratio = w_r_list[j]
        # run coop
        save_txt_weight = "jointEB_points_"+args.dataset+"/jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_dtW_"+args.dataset+".txt"
        save_txt_graph = "jointEB_points_"+args.dataset+"/jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_dtG_"+args.dataset+".txt"
        os.system("rm "+save_txt_weight)
        os.system("rm "+save_txt_graph)       
        first_d = -1 
        for total_epochs in range(99,100):
            # get times and g_epochs
            if total_epochs%4==0:
                times = 4
                g_epochs = total_epochs/4
            elif total_epochs%3==0:
                times = 3
                g_epochs = total_epochs/3
            elif total_epochs%2==0:
                times = 2
                g_epochs = total_epochs/2
            else:
                times = 1
                g_epochs = total_epochs
            g_epochs = int(g_epochs)
            times = int(times)
            model_name = "Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(total_epochs)+"_model.pth.tar"
            args.ratio_graph = g_ratio
            args.ratio_weight = w_ratio
            # args.times = times
            # args.epochs = g_epochs
            args.times = 4
            args.epochs = 25
            args.save_file = model_name
            os.system("rm "+"./prune_weight_iterate/"+model_name)

            ######### run repeatedly ##########
            dataset = args.dataset
            logging.basicConfig(filename=f"test_{dataset}_mask_change_even.txt",level=logging.DEBUG)
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
            dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
            # print(f"Number of graphs in {dataset} dataset:", len(dataset))
            data = dataset[0]
            model, data = Net(dataset, data, args).to(device), data.to(device)
            checkpoint = torch.load(f"./pretrain_pytorch/model.pth.tar")
            model.load_state_dict(checkpoint)

            loss = lambda m: F.nll_loss(m()[data.train_mask], data.y[data.train_mask])
            # print("construct admm training")
            support1 = model.adj1
            support2 = model.adj2
            partial_adj_mask = support1.clone()
            # print("num of edges * 2 + diag in adj:", np.count_nonzero(partial_adj_mask))
            adj_variables = [support1,support2]
            rho = 1e-3
            non_zero_idx = SparseTensor.from_torch_sparse_coo_tensor(model.adj1).nnz()
            Z1 = U1 = Z2 = U2 = partial_adj_mask.clone()
            model.adj1.requires_grad = True
            model.adj2.requires_grad = True
            adj_mask = partial_adj_mask.clone()


            id1 = model.id
            id2 = model.id
            # Define new loss function
            d1 = support1 + U1 - (Z1 + id1)
            d2 = support2 + U2 - (Z2 + id2)
            admm_loss = lambda m: loss(m) + \
                rho * (torch.sum(d1.coalesce().values() * d1.coalesce().values()) + 
                torch.sum(d2.coalesce().values()*d2.coalesce().values()))
            adj_optimizer = torch.optim.SGD(adj_variables,lr=0.001)
            weight_optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.01)
            adj_map = {"support1": support1, "support2": support2}

            # jteller@fiverings.com
            best_prune_acc = 0
            lookbacks = []
            counter = 0
            w_counter = 0
            pre3_mask1 = np.zeros((3703, 16))
            pre3_mask2 = np.zeros((16, 6))
            pre2_mask1 = np.zeros((3703, 16))
            pre2_mask2 = np.zeros((16, 6))
            pre1_mask1 = np.zeros((3703, 16))
            pre1_mask2 = np.zeros((16, 6))
            print('graph ratio:%d weight ratio:%d'%(g_ratio,w_ratio))
            for j in range(args.times):
                # warm up & prune weight
                for epoch in range(args.epochs):
                    print("times = %d, epoch = %d"%(times,epoch))
                    t_epoch = j*epoch
                    model.train()
                    weight_optimizer.zero_grad()
                    # Calculate gradient
                    admm_loss(model).backward(retain_graph=True)
                    weight_optimizer.step()

                    train_acc, val_acc, tmp_test_acc = test(model, data)
                    if val_acc > best_prune_acc:
                        best_prune_acc = val_acc
                        test_acc = tmp_test_acc
                    log = 'Pruning Time-Epoch: {:03d}-{:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    # print(log.format(j, epoch, train_acc, val_acc, tmp_test_acc))
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
                    thre_index = int(total * args.ratio_weight / 100)
                    thre = y[thre_index]
                    pruned = 0
                    # print('Pruning threshold: {}'.format(thre))
                    zero_flag = False
                    # print(model.conv1.weight.data)
                    for k, m in enumerate(model.modules()):
                        if isinstance(m, GCNConv):
                            weight_copy = m.weight.data.abs().clone()
                            mask = weight_copy.gt(thre).float().to(device)

                            mask_np = mask.cpu().numpy()
                            if(k==1):
                                current_mask1 = mask_np
                            elif(k==2):
                                current_mask2 = mask_np


                            pruned = pruned + mask.numel() - torch.sum(mask)
                            m.weight.data.mul_(mask)
                            if int(torch.sum(mask)) == 0:
                                zero_flag = True
                            # print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
                            #       format(k, mask.numel(), int(torch.sum(mask))))
                    # print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))
                    #######################
                    # print("\nTesting")
                    # print('current_mask1 = ',current_mask1)
                    # print('current_mask2 = ',current_mask2)
                    if (j==0 and epoch==0):
                        pre1_mask1 = current_mask1
                        pre1_mask2 = current_mask2
                    elif (j==0 and epoch==1):
                        pre2_mask1 = pre1_mask1
                        pre2_mask2 = pre1_mask2
                        pre1_mask1 = current_mask1
                        pre1_mask2 = current_mask2
                    elif (j==0 and epoch==2):
                        pre3_mask1 = pre2_mask1
                        pre3_mask2 = pre2_mask2
                        pre2_mask1 = pre1_mask1
                        pre2_mask2 = pre1_mask2
                        pre1_mask1 = current_mask1
                        pre1_mask2 = current_mask2
                    else:
                        dist_pre1_mask1 = calc_dist(pre1_mask1,current_mask1)
                        dist_pre1_mask2 = calc_dist(pre1_mask2,current_mask2)
                        dist_pre2_mask1 = calc_dist(pre2_mask1,current_mask1)
                        dist_pre2_mask2 = calc_dist(pre2_mask2,current_mask2)
                        dist_pre3_mask1 = calc_dist(pre3_mask1,current_mask1)
                        dist_pre3_mask2 = calc_dist(pre3_mask2,current_mask2)
                        dist_mask1 = np.max([dist_pre1_mask1,dist_pre2_mask1,dist_pre3_mask1])
                        dist_mask2 = np.max([dist_pre1_mask2,dist_pre2_mask2,dist_pre3_mask2])
                        total_dist = dist_mask1 + dist_mask2
                        total_dist = int(total_dist)
                        res_list.append(total_dist)  # 3,4,5,6,...,99
                        # print(res_list)
                        total_dist = int(total_dist)
                        f = open(save_txt_weight,'a')
                        print(total_dist,file=f)
                        f.close()
                        # if(g_ratio==80 and w_ratio==90 and t_epoch==99):
                        pre3_mask1 = pre2_mask1
                        pre3_mask2 = pre2_mask2
                        pre2_mask1 = pre1_mask1
                        pre2_mask2 = pre1_mask2
                        pre1_mask1 = current_mask1
                        pre1_mask2 = current_mask2
                    w_counter += 1

                # prune graph
                for epoch in range(args.epochs):
                    model.train()
                    adj_optimizer.zero_grad()
                    # Calculate gradient
                    admm_loss(model).backward(retain_graph=True)
                    # Update to correct gradient
                    update_gradients_adj(adj_map, adj_mask)
                    # Use the optimizer to update adjacency matrix
                    adj_optimizer.step()

                    train_acc, val_acc, tmp_test_acc = test(model, data)
                    if val_acc > best_prune_acc:
                        best_prune_acc = val_acc
                        test_acc = tmp_test_acc
                    log = "Pruning Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(j*args.epochs+epoch, train_acc, val_acc, tmp_test_acc)
                    cur_mask = get_mask(model.adj1 - id1, non_zero_idx, args.ratio_graph)    
                    if len(lookbacks) < args.lookback:
                        lookbacks.append(cur_mask)
                    else:
                        can_return = False            
                        total = 0
                        for mask in lookbacks:
                            dist = calc_dist(mask, cur_mask) 
                            if first_d == -1:
                                first_d = dist
                            dist /= first_d                            
                            total = max(dist,total)
                            if dist > args.thres:
                                can_return = False
                                # bre
                        logging.info(args.log.format(total)) # Here
                        total = int(total)
                        # print('counter = %d, total = %d'%(counter,total))
                        f = open(save_txt_graph,'a')
                        print(total,file=f)
                        f.close()
                        if can_return:
                            print(f"Found EB! At {j * args.epochs + epoch}")
                            post_processing()
                            exit()
                        lookbacks = lookbacks[1:]
                        lookbacks.append(cur_mask)
                    # torch.save(cur_mask, f"./masks/{args.dataset}_{args.ratio_graph}_{counter}_mask")
                    counter += 1
                    
                    # print(log.format(epoch, train_acc, best_prune_acc, test_acc))
                # Use learnt U1, Z1 and so on to prune
                adj1,adj2 = model.adj1, model.adj2
                Z1 = adj1 - id1 + U1
                Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
                U1 = U1 + (adj1 - id1 - Z1)

                Z2 = adj2 - id2 + U2
                Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
                U2 = U2 + (adj2 - id2 - Z2)

            adj1,adj2 = model.adj1, model.adj2
            adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
            adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
            model.adj1 = adj1.float()
            model.adj2 = adj2.float()

            # print("Optimization Finished!")
            train_acc, val_acc, tmp_test_acc = test(model, data)
            log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            # print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
            log_4_test = 'Tune Ratio: {:d}'
            # print(log_4_test.format(args.ratio))
            # cur_adj1 = model.adj1.cpu().numpy()
            # cur_adj2 = model.adj2.cpu().numpy()

            # torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./graph_pruned_pytorch/{args.save_file}")


        train_acc, val_acc, tmp_test_acc = test(model, data)
        log = 'After weight pruning: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(train_acc, val_acc, tmp_test_acc))

        # log_4_test = 'Graph Ratio: {:d}, Weight Ratio: {:d}, Test: {:.4f}'
        # print(log_4_test.format(args.ratio_graph, args.ratio_weight, tmp_test_acc))
        log_4_test = 'after retrain: Epoch: 99, Test: {:.4f}'
        # print(log_4_test.format(tmp_test_acc))

        # torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./prune_weight_iterate/{args.save_file}")




