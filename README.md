# CVPR2023-TWINS

Official code for "[TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization](https://placeholder)", CVPR 2023

## Requirement
PyTorch >= 1.9.0


## Description
The AT_cifar and AT_high_res_data are the baseline AT for CIFAR10/100 and high-resolution image data (Caltech, CUB, Stanford-Dogs). The proposed method is in TWINS_cifar and TWINS_high_res_data.


### Citation
If you use our code in your research, please cite with:

```
@inproceedings{
liu2023twins,
title={TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization},
author={Ziquan Liu and Yi Xu and Xiangyang Ji and Antoni B. Chan},
booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```

### Acknowledgement
We use [robustness](https://github.com/MadryLab/robustness) package in the robust model fine-tuning and [advprop](https://github.com/tingxueronghua/pytorch-classification-advprop) in the two-branch batch norm implementation.
