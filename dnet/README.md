# dnet
Deep Learning FrameWork for Urban Semantic Segmentation

"Deep Autoencoders with Aggregated Residual Transformations for Urban Reconstruction from Remote Sensing Data", CRV 2018

Authors: Timothy Forbes, Charalambos Poullis

Paper PDF: https://tinyurl.com/y73se9zs

Corresponding author: Timothy Forbes - timforby@gmail.com


## Folder structure

```bash
.
├── dnet
│   ├── models
│   │   ├── static
│   │   │   └── *.py
│   │   └── *.py
│   ├── handlers
│   │   └── *.py
│   ├── test.py
│   ├── train.py
│   └── README.md
├── data
│   ├── **DATASET**
│   │   ├── d_ng                    #DSM no_ground truth --Names must match rgb_d
│   │   │   └── *.png/*.tif/*.jpg
│   │   ├── rgb_ng                  #Imagery (RGB) no_ground truth
│   │   │   └── *.png/*.tif/*.jpg
│   │   ├── d                       #DSM with ground truth (y) --Names must match rgb
│   │   │   └── *.png/*.tif/*.jpg
│   │   ├── rgb                     #Imagery (RGB) with ground truth (y) 
│   │   │   └── *.png/*.tif/*.jpg
│   │   ├── y                       #Ground truth (RGB) y --Names must match rgb
│   │   │   └── *.png/*.tif/*.jpg
├── results
│   ├── **MODELNAME**
│   │   └── model.hdf5
```

---

## Run commands

> cd ./dnet

Before training must run mean.py on dataset.

### Training

> python train.py -i ../data/**DATASET** -o ../results -n **SOMEMODELNAME** -p **PATCHSIZE** -b **BATCHSIZE**

##### Continue training

> python train.py -i ../data/**DATASET** -o ../results -n **MODELNAME** --cont

### Testing

> python test.py -i ../data/**DATASET** -o ../results -n **MODELNAME**
