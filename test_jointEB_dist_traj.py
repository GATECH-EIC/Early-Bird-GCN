import os

dataset = "Cora"
prune_times = 100
update_epoch = int(100/prune_times)

g_r_list = [20,40,60,80]
w_r_list = [50,70,90]
for run_index in range(1,10+1):
    for i in range(4):
        for j in range(3):
            g_r = g_r_list[i]
            w_r = w_r_list[j]
            log_txt = "./test_jointEB_dist_traj_txts/times"+str(prune_times)+"/v"+str(run_index)+"/"+dataset+"_g"+str(g_r)+"_w"+str(w_r)+".txt"
            os.system("rm "+log_txt)
            os.system("rm ./masks/*")
            os.system("CUDA_VISIBLE_DEVICES=4 python3 run_threshold_jointEB.py"+" --times "+str(prune_times)+" --epochs "+str(update_epoch)+" --dataset "+dataset+" --ratio_graph "+str(g_r)+" --ratio_weight "+str(w_r)+" >>"+log_txt)


