# CLMTRcode

This is a pytorch implementation of the paper "CLMTR: A generic framework for contrastive multi-modal trajectory representation learning"

## Requirements
- Ubuntu 20.04.5 LTS with Python 3.9
- PyTorch 2.0.1
- torch-geometric
- Datasets can be downloaded from

## Quick Start
### Preprocessing
```bash
 python preprocessing.py --config 'configs/semantic_tdrive_process.yaml'
 python semantictraj_preprocessing.py --config 'configs/semantic_tdrive_process.yaml'
```
### Training
```bash
 python train.py --config 'configs/semantic_tdrive_batch256.yaml'
```

## FAQ
#### Datasets
To use your own datasets, you may need to create your own pre-processing script like `preprocessing.py`.