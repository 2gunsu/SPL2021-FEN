# FEN (Feature Enhancement Network)
Official implementation for our paper,  
**"Self-supervised Feature Enhancement Networks for Small Object Detection in Noisy Images"**.  
This paper has been accepted at **IEEE Signal Processing Letters 2021**.  
Currently, this paper is published in **Early Access area in IEEE Xplore** and can be viewed **[here](https://ieeexplore.ieee.org/document/9432743)**. 
  
**Authors**: *Geonsoo Lee*, *[Sungeun Hong](https://scholar.google.com/citations?user=CD27PpoAAAAJ&hl=ko&oi=ao)*, and *[Donghyeon Cho](https://scholar.google.com/citations?user=zj-NER4AAAAJ&hl=ko&oi=ao)*  
**Keywords**: Small Object Detection, Self-supervised Learning, Noisy Image  

## Requirements
**Note**: This code **cannot be executed** in **Windows** or **Multi-GPU** environment.  

Our code uses **[Detectron2](https://github.com/facebookresearch/detectron2)** developed by FAIR (Facebook AI Research).   
Therefore, please visit the repository and install the appropriate version that fits your environment.  

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
One is **DOTA** (for train and test) and the other is **ISPRS Toronto** (for only test).

### DOTA: A Large-scale Dataset for Object Detection in Aerial Images [[Paper](https://arxiv.org/abs/1711.10398)] [[Site](https://captain-whu.github.io/DOTA/dataset.html)]
You can download pre-processed DOTA dataset for our paper in this **[link](https://2gunsu.synology.me:1006/sharing/TCu337UJP)** directly.  
Please note that you can also download the raw dataset and pre-process it by yourself.  
The structure of the pre-processed data is as follows.  
Make sure ```Label.json``` follows the **[COCO data format](https://cocodataset.org/#format-data)**.

```
DOTA.zip
|-- Train
|   |-- Label.json
|   `-- Image
|       |-- Image_00000.png
|       |-- Image_00001.png
|       |-- Image_00002.png
|       `-- ...
|-- Test
|   |-- Label.json
|   `-- Image
|       |-- Image_00042.png
|       |-- Image_00055.png
|       |-- Image_00060.png
|       `-- ...
|-- Val
|   |-- Label.json
|   `-- Image
|       |-- Image_00066.png
|       |-- Image_00125.png
|       |-- Image_00130.png
|       `-- ...
`-- Mini
    |-- Label.json
    `-- Image
        |-- Image_00066.png
        |-- Image_00125.png
        |-- Image_00130.png
        `-- ...
```

### ISPRS Toronto [[Site](https://www.isprs.org/default.aspx)]
**Note**: This data cannot be used immediately due to its large resolution,  
and we will distribute the pre-processing code as soon as possible.  

(1) Please complete the data request form **[here](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/)**.  
(2) Access the FTP link you received by email.  
(3) Download all ```.tif``` image files in ```[FTP LINK]/ISPRS_BENCHMARK_DATASETS/Toronto/Images```.  
(4) Download the label files we made **[here](https://2gunsu.synology.me:1006/sharing/vzkqIH7kq)**. Like DOTA, these annotations also follow the coco data format.

## Usages
### Training
You can run ```run_train_net.py``` directly using IDEs like **Pycharm**.  
In this case, you have to manually fill in the required parameters in the code.  

You can also run ```run_train.py``` from the terminal with the command below.  

__Without FEN__
```python
python run_train.py --arch          [FILL]     # Select one in ['R50-FPN', 'R101-FPN', 'X101-FPN'] (Default: 'X101-FPN')
                    --data_root     [FILL]
                    --output_dir    [FILL]
                    --noise_type    [FILL]     # Select one in ['none', 'gaussian', 'snp'] (Default: 'none')
                    --noise_params  [FILL] 
                    --input_size    [FILL]     # Size of training data (Default: 800)
```

__With FEN__
```python
python run_train.py --arch          [FILL]     # Select one in ['R50-FPN', 'R101-FPN', 'X101-FPN']
                    --use_fen
                    --data_root     [FILL]
                    --output_dir    [FILL]
                    --noise_type    [FILL]     # Select one in ['none', 'gaussian', 'snp']
                    --noise_params  [FILL]
                    --fen_levels    [FILL]     # Make combinations using ['p2', 'p3', 'p4', 'p5', 'p6']
                                               # For example, ['p2', 'p4'], ['p5'], ['p3', 'p6'].
```


### Evaluation
You can run ```run_test_net.py``` directly using the IDE, or you can run ```run_test.py``` using the terminal.  
When using ```run_test.py```, the command is as follows.  
```python
python run_test.py --ckpt_root     [FILL]
                   --data_root     [FILL]
                   --noise_type    [FILL]     # Select one in ['none', 'gaussian', 'snp']
                   --noise_params  [FILL]             
```

## Qualitative Results
Column **(a)** is the base result when no method is applied, **(b)** and **(c)** are the results of applying **[Noise2Void](https://ieeexplore.ieee.org/document/8954066)** and **[DnCNN](https://ieeexplore.ieee.org/document/7839189)** at the **pixel** domain, respectively, and **(d)** is the result of applying our method at the **feature** domain.
<p align="center">
  <img src="/IMG/result_img.png" width="600" height="600">
</p>

## Quantitative Results
We have used five out of the standard evaluation metrics of **[COCO](https://cocodataset.org/#detection-eval)**.
<p align="center">
  <img src="/IMG/result_table.png">
</p>

## Citation
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
