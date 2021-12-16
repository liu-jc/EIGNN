# EIGNN: Efficient Infinite-Depth Graph Neural Networks
This repository is the official PyTorch implementation of "EIGNN: Efficient Infinite-Depth Graph Neural Networks" (NeurIPS 2021).


## Requirements
The script has been tested running under Python 3.6.9, with the following packages installed (along with their dependencies):
* pytorch (tested on 1.6.0)
* torch_geometric (tested on 1.6.3)
* scipy (tested on 1.5.2)
* numpy (tested on 1.19.2)

## Synthetic experiments
Here is an example to train the model on synthetic experiments: 

```
python train_EIGNN_chains.py --num_chains 20 --chain_len 100 --seed 14
```

`num_chains` and `chain_len` should be changed accordingly. 


## Experiments on real-world datasets 
### Evaluation and trained Models
We provide a few trained models on cornell datasets for demonstration purpose. 
See `saved_model`. 
To evaluate, run the command: 

```
python eval_EIGNN_heterophilic.py --dataset cornell --epoch 10000 --lr 0.8 --weight_decay 5e-06 --gamma 0.8 --idx_split 1 
```

`idx_split` should be changed accordingly. There are 10 data splits as used in [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn).

### Train the model 
Here is an example to train the model on university datasets, 

```
python train_EIGNN_heterophilic.py --dataset cornell --epoch 10000 --patience 500 --lr 0.8 --weight_decay 5e-06 --gamma 0.8 --idx_split 0 
```

This implementation is developed based on [the original implementation of IGNN](https://github.com/SwiftieH/IGNN). We thank them for their useful implementation.  

If you find our implementation useful in your research, please consider citing our paper:
```bibtex
@inproceedings{liu2021eignn,
 author = {Liu, Juncheng and Kawaguchi, Kenji and Hooi, Bryan and Wang, Yiwei and Xiao, Xiaokui},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {EIGNN: Efficient Infinite-Depth Graph Neural Networks},
 year = {2021}
}
```
