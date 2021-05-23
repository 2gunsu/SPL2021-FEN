# FEN (Feature Enhancement Network)
Official implementation for our paper,  
**"Self-supervised Feature Enhancement Networks for Small Object Detection in Noisy Images"**.  
This paper has been accepted at **IEEE Signal Processing Letters 2021**.  
Currently, this paper is published in **Early Access area in IEEE Xplore** and can be viewed [here](https://ieeexplore.ieee.org/document/9432743). 
  
**Authors**: *Geonsoo Lee*, *[Sungeun Hong](https://scholar.google.com/citations?user=CD27PpoAAAAJ&hl=ko&oi=ao)*, and *[Donghyeon Cho](https://scholar.google.com/citations?user=zj-NER4AAAAJ&hl=ko&oi=ao)*  
**Keywords**: Small Object Detection, Self-supervised Learning, Noisy Image  

## Requirements
**Note**: This code **cannot be executed** in Windows environment.  

Our code uses **[Detectron2](https://github.com/facebookresearch/detectron2)** developed by FAIR (Facebook AI Research).   
Therefore, please visit the repository and install the appropriate version for your environment.  

We have tested our code in the following environment.  
- **OS**: Ubuntu 18.04.5 LTS
- **GPU**: NVIDIA TITAN RTX (24 GB)
- **CUDA**: 11.0
- **Python**: 3.8.5
- **Pytorch**: 1.7.1
- **Torchvision**: 0.8.2
- **Detectron2**: 0.3

## Datasets and Preparation
In this paper, we have used two datasets.   
One is **DOTA** and the other is **ISPRS Torronto**.

#### (1) DOTA: A Large-scale Dataset for Object Detection in Aerial Images [[paper](https://arxiv.org/abs/1711.10398)]
You can download pre-processed dataset for our paper in this [link](https://2gunsu.synology.me:1006/fsdownload/eujZj9FTJ/DOTA.zip) directly.  
Please note that you can also download the raw dataset from [here](https://captain-whu.github.io/DOTA/dataset.html) and preprocess it yourself.

#### (2) ISPRS Toronto

## How to Use?
#### (1) Training
#### (2) Evaluation

## Qualitative Results
Column **(a)** is the base result when no method is applied, **(b)** and **(c)** are the results of applying [Noise2Void](https://ieeexplore.ieee.org/document/8954066) and [DnCNN](https://ieeexplore.ieee.org/document/7839189) at the pixel domain, respectively, and **(d)** is the result of applying our method at the feature domain.
<p align="center">
  <img src="/IMG/result_img.png" width="600" height="600">
</p>

## Quantitative Results
We have used five out of the standard evaluation metrics of [COCO](https://cocodataset.org/#detection-eval).
<p align="center">
  <img src="/IMG/result_table.png">
</p>

## Citation
This part will be updated later.
```
@ARTICLE{9432743,  
author={Lee, Geonsoo and Hong, Sungeun and Cho, Donghyeon},  
journal={IEEE Signal Processing Letters},   
title={Self-supervised Feature Enhancement Networks for Small Object Detection in Noisy Images},   
year={2021},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/LSP.2021.3081041}}
```
