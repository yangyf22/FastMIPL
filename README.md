# A PyTorch Implementation of FastMIPL

This is a PyTorch implementation of our paper "Fast Multi-Instance Partial-Label Learning"

```bibtex
@inproceedings{yang2024promipl,
    author = {Yin-Fang Yang and Wei Tang and Min-Ling Zhang},
    title = {Fast Multi-Instance Partial-Label Learning},
    booktitle = {Proceedings of the 39th {AAAI} Conference on Artificial Intelligence},
    year = {2025},
    address = {Philadelphia, Pennsylvania}
}
```

## Requirements

the file `environment.yml` records the requirement packages for this project.

To install the requirement packages, please run the following command:

```sh
conda env create -f environment.yml -n FastMIPL
```

Then, the environment can be activated by using the command

```sh
conda activate FastMIPL
```


## Datasets

Datasets used in this paper can be found on this [link](http://palm.seu.edu.cn/zhangml/Resources.htm#MIPL_data).



## Demo

To reproduce the results of MNIST_MIPL dataset in the paper, please run the following command:

```sh
python main.py --ds MNIST_MIPL --ds_suffix r1 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 40
python main.py --ds MNIST_MIPL --ds_suffix r2 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 30
python main.py --ds MNIST_MIPL --ds_suffix r3 --bs 350 --lr 5e-4 --epoch 200 --nr_samples 50
```


## Parameter Settings

| Datasets              | Learning Rate | Batch Size | Epochs    | nr_sampples   |
| --------------------- | ------------- | ---------- | --------- | ------------- |
| MNIST_MIPL (r = 1)    | 0.0005        | 350        | 200       | 40            |
| MNIST_MIPL (r = 2)    | 0.0005        | 350        | 200       | 30            |
| MNIST_MIPL (r = 3)    | 0.0005        | 350        | 200       | 50            |
| FMNIST_MIPL (r = 1)   | 0.0005        | 350        | 200       | 30            |
| FMNIST_MIPL (r = 2)   | 0.0005        | 350        | 200       | 40            |
| FMNIST_MIPL (r = 3)   | 0.0005        | 350        | 200       | 20            |
| Birdsong_MIPL (r = 1) | 0.001         | 910        | 500       | 10            |
| Birdsong_MIPL (r = 2) | 0.002         | 910        | 500       | 20            |
| Birdsong_MIPL (r = 3) | 0.005         | 910        | 500       | 30            |
| SIVAL_MIPL (r = 1)    | 0.002         | 1050       | 500       | 30            |
| SIVAL_MIPL (r = 2)    | 0.005         | 1050       | 500       | 20            |
| SIVAL_MIPL (r = 3)    | 0.005         | 1050       | 500       | 30            |
| CRC-MIPL-Row          | 0.001         | 4900       | 500       | 10            |
| CRC-MIPL-SBN          | 0.001         | 4900       | 500       | 10            |
| CRC-MIPL-KMeansSeg    | 0.001         | 4900       | 500       | 10            |
| CRC-MIPL-SIFT         | 0.001         | 4900       | 500       | 10            |


*N.B.*: This package is only free for academic usage.
