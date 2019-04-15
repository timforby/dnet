# dnet torch

Deep Learning FrameWork for Urban Semantic Segmentation

Uses slight variation of ScasNet: https://github.com/Yochengliu/ScasNet/

    @article{liu2018scasnet,   
      author = {Yongcheng Liu and    
                Bin Fan and    
                Lingfeng Wang and   
                Jun Bai and   
                Shiming Xiang and   
                Chunhong Pan},   
      title = {Semantic Labeling in Very High Resolution Images via A Self-Cascaded Convolutional Neural Network},   
      journal = {ISPRS J. Photogram. and Remote Sensing.},   
      volume = {145},  
      pages = {78--95},  
      year = {2018}   
    } 

---

## Procedure

> cd ./dnet_torch

This network trains using lists of images. To create the lists run "get_data_files.py"

### List creation

> python get_data_files.py -n **NAME OF DATASET (potsdam, singapore, etc)** -t **TYPE NAME (train, test, ground_truth, target)** -p **PATH TO IMAGES** -d **DIMENSIONS**

Note that for this project adding **-d** is necessary.

For this project, this must be done twice. 1st time for training images/imagery. 2nd time for ground truth/targets.

### Training

For this project I forgo arguments to the training file. Changes must be made by opening the file and editing the first couple lines of code.

Important changes are **train_paths**. List of train_paths must hold the files created with the "get_data_files.py".

> python train.py
