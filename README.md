# CLMTRcode

This is a pytorch implementation of the paper "CLMTR: A generic framework for contrastive multi-modal trajectory representation learning"

## Requirements
- Ubuntu 20.04.5 LTS 
- Python 3.9
- PyTorch 2.0.1 

## Quick Start
The datasets can be downloaded from [here](https://drive.google.com/file/d/1kntOZ5x9rpWzQtM9HrWUogxhAYxAsyx0/view?usp=drive_link), and tar -xzvf tdrive-samples.tar.gz

### Preprocessing
```bash
 python semantictraj_preprocessing.py --config 'configs/semantic_tdrive_process.yaml'
 python preprocessing.py --config 'configs/semantic_tdrive_process.yaml'
```

### Training

```bash
 python train.py --config 'configs/semantic_tdrive_batch256.yaml'
```

## FAQ
#### Datasets
To use your own datasets, you may need to create your own pre-processing script like `semantictraj_preprocessing.py` and `preprocessing.py`.
#### Installation
It may occur failure while installing torch-geometric related packages, including torch-scatter, torch-sparse, torch-cluster and torch-spline-conv, when using `pip install xxx` to install them directly. A solution can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 