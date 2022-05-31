# SyNoRiM - Jittor

### [**Paper**](https://arxiv.org/abs/2111.12878) | [**Project Page**](https://cg.cs.tsinghua.edu.cn/people/~huangjh/publication/synorim/)

This repository contains an experimental [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of 'SyNoRiM: Multiway Non-rigid Point Cloud Registration via Learned Functional Map Synchronization'.

Jittor is an open-source deep learning framework led by the computer graphics lab of Tsinghua University, under the supervision of [Prof. Shi-Min Hu](https://scholar.google.com/citations?user=LDb4tb0AAAAJ&hl=en). The two main contributions, i.e., **meta-operator fusion** and **uniform computation graph**, make it very fast, compared to traditional deep learning frameworks. Please join us via [this link](https://cg.cs.tsinghua.edu.cn/jittor/about/).

## Dependencies

The project mainly depends on `jittor`, please install it via:

```shell
pip install jittor
```

Other packages include: `numpy`, `omegaconf` and `tqdm`.

## Dataset

We provide MPC-DT4D dataset via this [Google Drive]() link. 
Please put the extracted data folder (i.e. `mpc-dt4d`) under `../dataset`, or modify the corresponding lines in `configs/data.yaml` to your downloaded location.
For other datasets and their respective configs, please refer to [here](https://github.com/huangjh-pub/synorim).

## Usage

To **train** our model, first you need to train a descriptor network:

```shell
python train.py configs/train_desc.yaml
```

The checkpoint will be stored automatically in `out/desc/` folder. Given a trained descriptor network, the bases are trained using:

```shell
python train.py configs/train_basis.yaml
```

To **test** the performance of the trained model, which is stored in `out/basis` folder, just run the evaluation script via:

```shell
python train.py configs/test.yaml
```

The jittor framework will use GPU automatically.

## Notice

The jittor implementation is largely based on [Jittor PointCloudLib](https://github.com/Jittor/PointCloudLib). We use PointNet++ to replace the sparse convolution used in the original paper. We also make several method-wise simplifications to adapt to jittor. For evaluation please stick to the [original version](https://github.com/huangjh-pub/synorim/tree/public).

## Citation

```bibtex
@article{huang2021multiway,
  title={Multiway Non-rigid Point Cloud Registration via Learned Functional Map Synchronization},
  author={Huang, Jiahui and Birdal, Tolga and Gojcic, Zan and Guibas, Leonidas J and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2111.12878},
  year={2021}
}
```
