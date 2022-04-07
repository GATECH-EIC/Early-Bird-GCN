# Early-Bird GCNs: Graph-Network Co-Optimization Towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

Haoran You, Zhihan Lu, Zijian Zhou, Yonggan Fu, Yingyan Lin

Accepted by [AAAI 2022](https://aaai.org/Conferences/AAAI-22/). More Info:
\[ [**Paper**](https://www.aaai.org/AAAI22Papers/AAAI-6732.YouH.pdf) | [**Appendix**](https://github.com/RICE-EIC/Early-Bird-GCN/blob/main/2022AAAI_EB_GCN_Supple.pdf) | [**Slide**](https://drive.google.com/file/d/1WcJzOaoXIvJzg063U1HOzNqhoRNYJaCM/view?usp=sharing) | [**Poster**](https://drive.google.com/file/d/1VHK--X6nDytoAxl6eY0Nlws8q2ls54kV/view?usp=sharing) | [**Video**](https://slideslive.com/38976676/earlybird-gcns-graphnetwork-cooptimization-towards-more-efficient-gcn-training-and-inference-via-drawing-earlybird-lottery-tickets) | [**Github**](https://github.com/RICE-EIC/Early-Bird-GCN) \]



### Install the conda environment

```shell
conda env create -f env.yaml
pip install torch_geometric
pip uninstall torch-scatter
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip uninstall torch-sparse
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip uninstall torch-cluster
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip uninstall torch-spline-conv
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
```

### Run the code

- To pretrain, prune, retrain separately: 

  - Pretrain the GCN: 

  - ```shell
    python3 pytorch_train.py --epochs 10 --dataset Cora
    ```

  - Prune the pretrained GCN using different prune method: 

  - ```shell
    python3 pytorch_prune_weight_iterate.py --ratio_graph 60 --ratio_weight 60
    # or
    python3 pytorch_prune_weight_cotrain.py --ratio_graph 60 --ratio_weight 60
    # or 
    python3 pytorch_prune_weight_first.py --ratio_graph 60 --ratio_weight 60
    ```

  - Retrain the pruned GCN to recover the accuracy: 

  - ```shell
    python3 pytorch_retrain_with_graph.py --load_path prune_weight_iterate/model.pth.tar
    ```

- By using functions like ```os.system("python3 "+"pytorch_train.py"+" --epochs "+str(1)+" --dataset "+str(args.dataset))``` in Python, we are able to run the above process in one file, and can stop automatically when found jointEB ticket: 

  - ```shell
    python run_threshold_jointEB.py --times 100 --epochs 1 --dataset Cora --ratio_graph 20 --ratio_weight 50
    ```

- Futhermore, we use a script to run all experiment settings (like different pruning ratio of graph and pruning ratio of weights) automatically: 

  - ```
    python test_jointEB_dist_traj.py
    ```

### Run the code with Sparse Graph, SGCN-deep stuff

More details are coming soon.
