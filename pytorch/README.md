# SyNoRiM: Multiway Non-rigid Point Cloud Registration via Learned Functional Map Synchronization

### [**Paper**](https://arxiv.org/abs/2111.12878) | [**Project Page**](https://cg.cs.tsinghua.edu.cn/people/~huangjh/publication/synorim/)

![](./assets/teaser.png)

This repository contains the implementation of the above paper. It is accepted to **T-PAMI (2022)**.
- Authors: [Jiahui Huang](https://cg.cs.tsinghua.edu.cn/people/~huangjh/), [Tolga Birdal](http://tbirdal.me/), [Zan Gojcic](https://zgojcic.github.io/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/), [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)
    - Contact Jiahui either via email or github issues :)


If you find our code or paper useful, please consider citing
```bibtex
@article{huang2021multiway,
  title={Multiway Non-rigid Point Cloud Registration via Learned Functional Map Synchronization},
  author={Huang, Jiahui and Birdal, Tolga and Gojcic, Zan and Guibas, Leonidas J and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2111.12878},
  year={2021}
}
```

## Installation
We suggest to use [Anaconda](https://www.anaconda.com/) to manage your environment. Following is the suggested way to install the dependencies:

```bash
# Create a new conda environment
conda create -n synorim python=3.8
conda activate synorim

# Install pytorch
conda install pytorch==1.10.2 cudatoolkit=11.1 -c pytorch -c nvidia

# Install other packages (ninja is required for MinkowskiEngine)
pip install -r requirements.txt
```

Note: You can also choose to use our `MinkowskiEngine` fork from [here](https://github.com/heiwang1997/MinkowskiEngine), with a RAM leak bug fixed.

## Dataset

| Name       | # Train / Val. | # Test | Term of Use                                                  | Downloads                                                    | L2 Error |
| ---------- | -------------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| MPC-CAPE   | 3015 / 798     | 209    | [Link](https://cape.is.tue.mpg.de/license.html) ⚠️            | [Data (11.2G)](https://drive.google.com/file/d/1der12IAm_1o_M92nj71r0HpfxmBCaQmc/view?usp=sharing)   Models: [Full](https://drive.google.com/file/d/1FMgXeM8zX448j8zQulfm0Zy0aHfblRPJ/view?usp=sharing) [Self](https://drive.google.com/file/d/11ohjDuFkujYDDrI-XD8YP22DSO2cd5Au/view?usp=sharing) | 1.08 / 2.93     |
| MPC-DT4D   | 3907 / 1701    | 1299   | [Link](https://docs.google.com/forms/d/e/1FAIpQLSckMLPBO8HB8gJsIXFQHtYVQaTPTdd-rZQzyr9LIIkHA515Sg/viewform) | [Data (20.6G)](https://drive.google.com/file/d/1r9VFHIZcatSej6guY_hGoGjrNqbgazAz/view?usp=sharing)   [Model](https://drive.google.com/file/d/1vs9rOfGeXOXk6Q4gGfkDR3ziCV_XvKYT/view?usp=sharing) | 3.53     |
| MPC-DD     | 1754 / 200     | 267    | [Link](https://docs.google.com/forms/d/e/1FAIpQLSeQ1hkCmmTiib-oQM9s21y3Tz9ojiI2zB8vZSqTZjT2DiRZ0g/viewform) | [Data (2.4G)](https://drive.google.com/file/d/1ykFSe9TI9kZ-RozZw874YHDiO1cLRCgc/view?usp=sharing)   [Model](https://drive.google.com/file/d/1ezvt-MoW0AIMHiJj8j9RZFu2PQWPqaLw/view?usp=sharing) | 2.54     |
| MPC-SAPIEN | 530 / 88       | 266    | [Link](https://sapien.ucsd.edu/about#term)                   | [Data (1.3G)](https://drive.google.com/file/d/13yMOoFmUV2Ca9j0tm_CD0nd1BGx1T8Jx/view?usp=sharing)   [Model](https://drive.google.com/file/d/1mmJDrVsDbUd1wjazDKGpeF0tUrpYtUDu/view?usp=sharing) | 3.05     |

We provide all the datasets used in our paper. However, **in order to download them, please accept the respective 'Term of Use' by sending the agreement form if needed**. The data are stored on Google Drive. Please use a VPN if you cannot access it directly.

You can also run the following script to download all datasets and automatically place them in the correct place (`../dataset`). Please make sure you have write access to that folder because it is outside of the project folder.

```bash
bash scripts/download_datasets.sh
```
Each dataset is organized in the following structure.

```
<dataset-name>/
    ├ meta.json		# Specifies train/val/test split and #frames in each npz file
    └ data/
        ├ 0000.npz	# Each npz file contains point clouds and all pairwise flows.
        ├ 0001.npz	# Please refer to 'dataset/flow_dataset.py' to see how to parse them.
        └ ...
```

## Usage

### Pre-trained model

We provide pretrained models for all the datasets. You can either download them from the links in the table above, or use the following script:

```shell
bash scripts/download_pretrained.sh
```

If you manually download the checkpoints, put them into `out_pretrained` folder.

You can then run the evaluation script to reproduce our metrics:

```shell
python evaluate.py configs/<DATASET-NAME>/test_pretrained.yaml
```

### Training

Our full training procedure is divided into two steps. First, train the descriptor network using:

```shell
python train.py configs/<DATASET-NAME>/train_desc.yaml
```

After the training converges, train the basis network with:

```shell
python train.py configs/<DATASET-NAME>/train_basis.yaml
```

The script will log config files, models, and tensorboard logs into `out` folder. You may need to use tensorboard to monitor the training process:

```
tensorboard --logdir=out
```

Please refer to all the yaml files for available training options. We use a inclusion mechanism so also check out the yaml files linked in `include_configs`.

### Evaluation

The full evaluation process takes multiple input point clouds and run synchronization algorithm introduced in our paper. Please run the following script to evaluate a trained model.
```shell
python evaluate.py configs/<DATASET-NAME>/test.yaml
```
*Note*: after the basis training phase is completed, the weights in the descriptor network are included in the basis checkpoint. However the config file for the descriptor network is still needed to recover meta-data. 
