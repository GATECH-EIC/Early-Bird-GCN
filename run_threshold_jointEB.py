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
import logging
import os
import random

def random_label(data):
    # print("shuffling graph's label... ")
    labels = data.y.numpy()
    # print("label shape = ",labels.shape) #(3327,)
    # print("label max = ",np.max(labels)) #5
    # print("label min = ",np.min(labels)) #0
    node_num = labels.shape[0]
    labels_cnt = np.zeros((np.max(labels)+1))
    for i in range(np.min(labels),np.max(labels)+1):
        labels_cnt[i] = np.count_nonzero(labels==i)
    labels_cnt = labels_cnt.astype(np.int16)
    # print(labels)
    # print("labels shape",labels.shape)
    # print(labels_cnt) #[264 590 668 701 596 508]
    randomed_labels = np.zeros((node_num)) #(3327)
    for i in range(np.min(labels)+1,np.max(labels)+1): #[1,5]
        for j in range(labels_cnt[i]):
            random_node_id = random.randint(0,node_num-1)
            while(randomed_labels[random_node_id]!=0):
                random_node_id = random.randint(0,node_num-1)
            randomed_labels[random_node_id]=i
    randomed_labels = randomed_labels.astype(np.int16)

    for i in range(np.min(randomed_labels),np.max(randomed_labels)+1):
        labels_cnt[i] = np.count_nonzero(randomed_labels==i)
    labels_cnt = labels_cnt.astype(np.int16)
    # print(randomed_labels)
    # print("randomed_labels shape",randomed_labels.shape)
    # print(labels_cnt) #[264 590 668 701 596 508]
    data.y = torch.from_numpy(randomed_labels).long()
    # print("shuffling done! ")
    return data
def half_dataset(data):
    # print("half dataset...")
    # print("data = ",data) # Data(edge_index=[2, 9104], test_mask=[3327], train_mask=[3327], val_mask=[3327], x=[3327, 3703], y=[3327])
    train_mask = data.train_mask.numpy() #120 [0,119]
    # test_mask = data.test_mask.numpy() #1000 [2312,3326]
    # val_mask = data.val_mask.numpy() #500 [120,619]
    train_num = np.count_nonzero(train_mask==True)
    train_mask = np.zeros((train_mask.shape[0]))
    for i in range(int(train_num)):
        if(i<int(train_num/2)):
            train_mask[i] = True
        else:
            train_mask[i] = False
    # print(np.count_nonzero(train_mask==True)) #60 [0,59]
    data.train_mask = torch.from_numpy(train_mask).bool()
    # print("half dataset done!")
    return data
def nodeid_shuffled(data):
    # print("shuffle dataset...")
    # print("data = ",data) # Data(edge_index=[2, 9104], test_mask=[3327], train_mask=[3327], val_mask=[3327], x=[3327, 3703], y=[3327])
    node_num = data.train_mask.shape[0]
    node_id_map = random.sample(range(node_num),node_num)
    # change node id edge 
    edge_index = data.edge_index.numpy() #120 [0,119]
    for i in range(edge_index.shape[0]): 
        for j in range(edge_index.shape[1]):
            edge_index[i][j] = node_id_map[edge_index[i][j]]
    data.edge_index = torch.from_numpy(edge_index).long()
    # change test mask
    test_mask = data.test_mask.numpy()
    new_test_mask = np.zeros(test_mask.shape)
    for i in range(test_mask.shape[0]):
        if (test_mask[i]==True):
            new_test_mask[node_id_map[i]] = True
    data.test_mask = torch.from_numpy(new_test_mask).bool()
    # change train mask
    train_mask = data.train_mask.numpy()
    new_train_mask = np.zeros(train_mask.shape)
    for i in range(train_mask.shape[0]):
        if (train_mask[i]==True):
            new_train_mask[node_id_map[i]] = True
    data.train_mask = torch.from_numpy(new_train_mask).bool()
    # change val mask
    val_mask = data.val_mask.numpy()
    new_val_mask = np.zeros(val_mask.shape)
    for i in range(val_mask.shape[0]):
        if (val_mask[i]==True):
            new_val_mask[node_id_map[i]] = True
    data.val_mask = torch.from_numpy(new_val_mask).bool()
    # change node feature
    features = data.x.numpy() # data.x: [node_num, node_feature]
    new_features = np.zeros(features.shape)
    for i in range(features.shape[0]):
        new_features[node_id_map[i]] = features[i]
    new_features = new_features.astype(np.float32)
    data.x = torch.from_numpy(new_features)
    # change node label
    labels = data.y.numpy()
    map_labels = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]): # i is node id
        map_labels[node_id_map[i]] = labels[i]
    map_labels = map_labels.astype(np.int16)
    data.y = torch.from_numpy(map_labels).long()
    return data
def layerwise_rearrange(data): # m.weight.data
    new_data = data.cpu().numpy()
    np.random.shuffle(new_data)
    new_data = torch.from_numpy(new_data)
    return new_data

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
            transposed_temp_grad_adj = torch.transpose(temp_grad_adj,1,0)
            temp_grad_adj1 = temp_grad_adj + transposed_temp_grad_adj
            var1 = var
        if key == "support2":
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
def get_mask(oriadj:torch.Tensor, non_zero_idx:int, percent:int) -> torch.Tensor:
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
    new_adj = low_adj + np.transpose(low_adj)
    new_adj = np.add(new_adj, np.identity(new_adj.shape[0]))
    return 1 - (new_adj != adj)

def calc_dist(m1,m2):
    return np.abs(m1 - m2).sum()

def post_processing():
    # print("here in post_processing")
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
    # torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./graph_pruned_eb_pytorch/{args.save_file}")
parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--ratio_graph', type=int, default=0)
parser.add_argument('--ratio_weight', type=int, default=90)
parser.add_argument('--use_gdc', type=bool, default=False)
parser.add_argument('--save_file', type=str, default="model.pth.tar")
parser.add_argument('--lookback', type=int, default=3)
parser.add_argument("--thres", type=float, default=0.1)
parser.add_argument("--dataset", type=str, default="Pubmed")
parser.add_argument("--log", type=str, default="{:05d}")
parser.add_argument("--is_random_label", type=int, default=0)
parser.add_argument("--is_half_dataset", type=int, default=0)
parser.add_argument("--is_nodeid_shuffled", type=int, default=0)
parser.add_argument("--is_layerwise_rearrange", type=int, default=0)
parser.add_argument("--is_ticket", type=int, default=-1) # -1 means GEBT. [0,20] means other tickets. Initial ticket = 0, partially trained = 49
parser.add_argument("--is_random_prune", type=int, default=0)
parser.add_argument("--is_smart_ratio", type=int, default=0)
parser.add_argument("--is_need_thres", type=int, default=0)
parser.add_argument("--is_need_retrain_acc", type=int, default=0)

args = parser.parse_args()

g_ratio = args.ratio_graph
w_ratio = args.ratio_weight
models = ["pruned_pytorch/model.pth.tar","prune_weight_cotrain/model.pth.tar","prune_weight_iterate/model.pth.tar","prune_weight_first/model.pth.tar"]
res_list = []
g_r_list = [20,40,60,80]
w_r_list = [50,70,90]
# test without pretrain, prune from scratch
pretrain_model_name = "./pretrain_pytorch/"+str(args.dataset)+"_model.pth.tar"
os.system("rm "+pretrain_model_name)
os.system("python3 "+"pytorch_train.py"+" --epochs "+str(1)+" --dataset "+str(args.dataset))
# run coop and find joint EB
exit_flag = 0
jEB = 100

dataset = args.dataset
logging.basicConfig(filename=f"test_{dataset}_mask_change_even.txt",level=logging.DEBUG)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures()) # CiteSeer()
# print(f"Number of graphs in {dataset} dataset:", len(dataset)) # len=1
data = dataset[0] # CiteSeer: Data(edge_index=[2, 9104], test_mask=[3327], train_mask=[3327], val_mask=[3327], x=[3327, 3703], y=[3327])
if(args.is_random_label==1):
    data = random_label(data)
if(args.is_half_dataset==1):
    data = half_dataset(data)
if(args.is_nodeid_shuffled==1):
    data = nodeid_shuffled(data)



model, data = Net(dataset, data, args).to(device), data.to(device)
checkpoint = torch.load(pretrain_model_name)
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
adj_mask = torch.from_numpy(partial_adj_mask).to(device)
id1 = torch.eye(support1.shape[0]).to(device)
id2 = torch.eye(support2.shape[0]).to(device)
# Define new loss function
admm_loss = lambda m: loss(m) + \
            rho * (F.mse_loss(support1 + U1, Z1 + id1) +
            F.mse_loss(support2 + U2, Z2 + id2))
adj_optimizer = torch.optim.Adam(adj_variables,lr=0.001)
weight_optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)
adj_map = {"support1": support1, "support2": support2}

best_prune_acc = 0
lookbacks = []
counter = 0
pre3_mask1 = np.zeros((3703, 16))
pre3_mask2 = np.zeros((16, 6))
pre2_mask1 = np.zeros((3703, 16))
pre2_mask2 = np.zeros((16, 6))
pre1_mask1 = np.zeros((3703, 16))
pre1_mask2 = np.zeros((16, 6))
weight_norm_baseline = -1
graph_norm_baseline = -1
total_dist = 0
graph_dist = 0
print('times:%3d epochs:%3d dataset:%10s graph ratio:%2d weight ratio:%2d'%(args.times,args.epochs,args.dataset,g_ratio,w_ratio))

# start pruning iterately, since the times*epoch is uncertain, set epoch = 1, just change times, 
# which means graph will be pruned more frequently, the graph dist will change more dramatically. 


epoch_cnt = -1
saved_ticket_weights = np.zeros((1,1))
for time in range(args.times):
    for update_epoch in range(args.epochs):
        epoch_cnt += 1
        #STEP1: warm up & update weight optimizer
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

        #STEP2: prune weight & compute total_dist
        total = 0
        total_layer1 = 0
        total_layer2 = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, GCNConv):
                total += m.weight.data.numel() # number of elements
                if(k==1):
                    total_layer1 += m.weight.data.numel()
                elif(k==2):
                    total_layer2 += m.weight.data.numel()

        conv_weights = torch.zeros(total)
        ticket_conv_weights = torch.zeros(total)
        layer1_conv_weights = torch.zeros(total_layer1)
        layer2_conv_weights = torch.zeros(total_layer2)
        index = 0
        for k, m in enumerate(model.modules()):
            if isinstance(m, GCNConv):
                if(k==1):
                    layer1_conv_weights[0:total_layer1] = m.weight.data.view(-1).abs().clone()
                elif(k==2):
                    layer2_conv_weights[0:total_layer2] = m.weight.data.view(-1).abs().clone()
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                ticket_conv_weights[index:(index + size)] = m.weight.data.view(-1).clone()
                index += size
                
        y, i = torch.sort(conv_weights) # get the weight's global prioroty. y:sorted matrix i:index
        y1, i1 = torch.sort(layer1_conv_weights)
        y2, i2 = torch.sort(layer2_conv_weights)
        thre_index = int(total * args.ratio_weight / 100) # number of weights to be pruned
        thre = y[thre_index] # get threshold value

        thre_index1 = int(total_layer1 * args.ratio_weight / 100 / 4) # number of weights to be pruned
        thre1 = y1[thre_index1] # get threshold value
        thre_index2 = int(total_layer2 * args.ratio_weight * 3 / 100 / 4) # number of weights to be pruned
        thre2 = y2[thre_index2] # get threshold value


        pruned = 0
        # print('Pruning threshold: {}'.format(thre))
        zero_flag = False
        # print(model.conv1.weight.data)

        # save ticket's weight
        if(args.is_ticket!=-1 and epoch_cnt==args.is_ticket):
            # print("saved ticket's weights at epoch ",epoch_cnt)
            saved_ticket_weights = ticket_conv_weights
            # print("saved_ticket_weights shape = ",saved_ticket_weights.shape)
        
        # if use random prune and smart ratio
        if(args.is_random_prune==1 and args.is_smart_ratio==1): # shuffle them and prune front p%
            for k, m in enumerate(model.modules()):
                if isinstance(m, GCNConv):
                    # do layerwise weight rearrange
                    weight_data = m.weight.data
                    weight_data = layerwise_rearrange(weight_data)
                    m.weight.data = weight_data

                    weight_copy = m.weight.data.abs().clone()
                    mask = np.zeros(weight_copy.cpu().numpy().shape)
                    if(k==1):
                        mask = mask.reshape(-1)
                        for index in range(thre_index1):
                            mask[index] = 1
                        mask = mask.reshape(weight_copy.cpu().numpy().shape)
                        mask = torch.from_numpy(mask).float().to(device) # get prune mask
                        mask_np = mask.cpu().numpy()
                        current_mask1 = mask_np
                        ticket_final_mask1 = mask
                    elif(k==2):
                        mask = mask.reshape(-1)
                        for index in range(thre_index2):
                            mask[index] = 1
                        mask = mask.reshape(weight_copy.cpu().numpy().shape)
                        mask = torch.from_numpy(mask).float().to(device) # get prune mask
                        mask_np = mask.cpu().numpy()
                        current_mask2 = mask_np
                        ticket_final_mask2 = mask
                    m.weight.data.mul_(mask) # prune weight through mask
        elif(args.is_random_prune!=1 and args.is_smart_ratio==1):
            for k, m in enumerate(model.modules()):
                if isinstance(m, GCNConv):
                    if(k==1):
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(thre1).float().to(device) # if smaller than thre, set 0
                        mask_np = mask.cpu().numpy()
                        current_mask1 = mask_np
                        ticket_final_mask1 = mask
                    elif(k==2):
                        weight_copy = m.weight.data.abs().clone()
                        mask = weight_copy.gt(thre2).float().to(device) # if smaller than thre, set 0
                        mask_np = mask.cpu().numpy()
                        current_mask2 = mask_np
                        ticket_final_mask2 = mask
                    m.weight.data.mul_(mask) # prune weight through mask
        else:
            for k, m in enumerate(model.modules()):
                if isinstance(m, GCNConv):
                    # do layerwise weight rearrange
                    if(args.is_layerwise_rearrange==1):
                        weight_data = m.weight.data
                        weight_data = layerwise_rearrange(weight_data)
                        m.weight.data = weight_data
                    weight_copy = m.weight.data.abs().clone()
                    mask = weight_copy.gt(thre).float().to(device) # if smaller than thre, set 0
                    mask_np = mask.cpu().numpy()
                    if(k==1):
                        current_mask1 = mask_np
                        ticket_final_mask1 = mask
                    elif(k==2):
                        current_mask2 = mask_np
                        ticket_final_mask2 = mask
                    m.weight.data.mul_(mask) # prune weight through mask

        if (epoch_cnt==0):
            pre1_mask1 = current_mask1
            pre1_mask2 = current_mask2
        elif (epoch_cnt==1):
            pre2_mask1 = pre1_mask1
            pre2_mask2 = pre1_mask2
            pre1_mask1 = current_mask1
            pre1_mask2 = current_mask2
        elif (epoch_cnt==2):
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
            # print('total_dist_before = ',total_dist)
            if (weight_norm_baseline==-1 or weight_norm_baseline==0):
                weight_norm_baseline = total_dist # set the first total_dist value to be norm 1
            # print('weight_norm_baseline = ',weight_norm_baseline)
            total_dist /= weight_norm_baseline
            pre3_mask1 = pre2_mask1
            pre3_mask2 = pre2_mask2
            pre2_mask1 = pre1_mask1
            pre2_mask2 = pre1_mask2
            pre1_mask1 = current_mask1
            pre1_mask2 = current_mask2

        #STEP3: update graph optimizer & compute graph dist
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
        cur_mask = get_mask(model.adj1 - id1, non_zero_idx, args.ratio_graph)    
        if len(lookbacks) < args.lookback:
            lookbacks.append(cur_mask)
        else:
            can_return = False            
            total = 0
            for mask in lookbacks:
                dist = calc_dist(mask, cur_mask) / cur_mask.size  
                total = max(calc_dist(mask, cur_mask),total)
                if dist > args.thres:
                    can_return = False
                    # bre
            logging.info(args.log.format(total)) # Here
            # print('total_before = ',total)
            if(graph_norm_baseline==-1 or graph_norm_baseline==0):
                graph_norm_baseline = total
            # print('graph_norm_baseline = ',graph_norm_baseline)
            total /= graph_norm_baseline
            graph_dist = total
            lookbacks = lookbacks[1:]
            lookbacks.append(cur_mask)
        torch.save(cur_mask, f"./masks/{args.dataset}_{args.ratio_graph}_{counter}_mask")
        counter += 1
        #STEP4: update U,Z
        adj1,adj2 = model.adj1, model.adj2
        Z1 = adj1 - id1 + U1
        Z1 = prune_adj(Z1,non_zero_idx,args.ratio_graph) - id1
        U1 = U1 + (adj1 - id1 - Z1)
        Z2 = adj2 - id2 + U2
        Z2 = prune_adj(Z2,non_zero_idx,args.ratio_graph) - id2
        U2 = U2 + (adj2 - id2 - Z2)
        #STEP5: compute joint value
        if(args.ratio_graph==0):
            joint_value = total_dist
        elif(args.ratio_weight==0):
            joint_dist = graph_dist
        else:
            joint_value = np.mean([total_dist,graph_dist])
        # print('epoch = %2d, w_dist = %.3f, g_dist = %.3f, joint_dist = %.3f'%(epoch_cnt,total_dist,graph_dist,joint_value))
        
        #STEP6: find jointEB
        if(args.is_need_thres==1 and epoch_cnt>5 and joint_value<args.thres):
            # recover to ticket's weight at final epoch
            if(args.is_ticket!=-1):
                index = 0
                for m in model.modules():
                    if isinstance(m, GCNConv):
                        size = m.weight.data.numel()
                        m.weight.data = saved_ticket_weights[index:(index + size)].view(m.weight.data.shape).clone().to(device)
                        index += size
                # apply previous mask
                for k, m in enumerate(model.modules()):
                    if isinstance(m, GCNConv):
                        # do layerwise weight rearrange
                        if(args.is_layerwise_rearrange==1):
                            weight_data = m.weight.data
                            weight_data = layerwise_rearrange(weight_data)
                            m.weight.data = weight_data
                        if(k==1):
                            m.weight.data.mul_(ticket_final_mask1) # prune weight through mask
                        elif(k==2):
                            m.weight.data.mul_(ticket_final_mask2) # prune weight through mask
                # print("recoverd saved ticket's weights at final epoch")

            print('EB found! thres = %.2f, current epoch:%2d'%(args.thres, epoch_cnt))
            # prune graph
            adj1,adj2 = model.adj1, model.adj2
            adj1 = prune_adj(adj1 - id1, non_zero_idx, args.ratio_graph)
            adj2 = prune_adj(adj2 - id2, non_zero_idx, args.ratio_graph)
            model.adj1 = adj1
            model.adj2 = adj2
            # test acc
            train_acc, val_acc, tmp_test_acc = test(model, data)
            log = 'After tune results: Ratio: {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            # print(log.format(args.ratio, train_acc, val_acc, tmp_test_acc))
            log_4_test = 'Tune Ratio: {:d}'
            # print(log_4_test.format(args.ratio))
            cur_adj1 = model.adj1.cpu().numpy()
            cur_adj2 = model.adj2.cpu().numpy()
            jEB = epoch_cnt
            model_name = "jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(jEB)+"_model.pth.tar"
            torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./jointEB_pruned_pytorch/"+model_name)
            exit_flag = 1
        elif(args.is_need_thres==0):
            exit_flag = 0
        if(exit_flag==1):
            break
    if(exit_flag==1):
            break

if(args.is_ticket!=-1 and exit_flag==0): # recover to ticket's weight at final epoch
    index = 0
    for m in model.modules():
        if isinstance(m, GCNConv):
            size = m.weight.data.numel()
            m.weight.data = saved_ticket_weights[index:(index + size)].view(m.weight.data.shape).clone().to(device)
            index += size
    # apply previous mask
    for k, m in enumerate(model.modules()):
        if isinstance(m, GCNConv):
            # do layerwise weight rearrange
            if(args.is_layerwise_rearrange==1):
                weight_data = m.weight.data
                weight_data = layerwise_rearrange(weight_data)
                m.weight.data = weight_data
            if(k==1):
                m.weight.data.mul_(ticket_final_mask1) # prune weight through mask
            elif(k==2):
                m.weight.data.mul_(ticket_final_mask2) # prune weight through mask
    # print("recoverd saved ticket's weights at final epoch")
    jEB = 0
    cur_adj1 = model.adj1.cpu().detach().numpy()
    model_name = "jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(jEB)+"_model.pth.tar"
    torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./jointEB_pruned_pytorch/"+model_name)
    exit_flag = 1

if(args.times==0 and args.epochs==0):
    jEB = 0
    cur_adj1 = model.adj1.cpu().detach().numpy()
    model_name = "jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(jEB)+"_model.pth.tar"
    torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./jointEB_pruned_pytorch/"+model_name)
    exit_flag = 1
if(args.is_smart_ratio==1):
    jEB = 0
    cur_adj1 = model.adj1.cpu().detach().numpy()
    model_name = "jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(jEB)+"_model.pth.tar"
    torch.save({"state_dict":model.state_dict(),"adj":cur_adj1}, f"./jointEB_pruned_pytorch/"+model_name)
    exit_flag = 1
# retrain to test jointEB acc
if(exit_flag==1 and args.is_need_retrain_acc==1):
    # print("test retrain acc#######")
    model_name = "jointEB_pruned_pytorch/"+"jointEB_Gr"+str(g_ratio)+"_Wr"+str(w_ratio)+"_E"+str(jEB)+"_model.pth.tar"
    os.system("python3 "+"pytorch_retrain_with_graph.py"+" --load_path "+model_name+" --dataset "+str(args.dataset))







