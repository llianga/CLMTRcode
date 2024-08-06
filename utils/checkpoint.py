import os
import logging
from typing import Optional

import torch

from config import Config

class CheckPointer(object):
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: Optional[torch.optim.Optimizer]=None, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, 
                 save_dir: str=""):
        super(CheckPointer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.logger = logging.getLogger('CL')

    def save(self, epoch_idx, rank, vali_loss, tag=""):
        name =f"{Config.DATASETS.dataset}_{Config.distance_type}_epoch_{str(epoch_idx)}_rank_{rank}_Loss_{vali_loss}.pt"
        #i_ep, recall_1, recall_5, recall_10_50, vali_loss
        # name =f"{Config.DATASETS.dataset}_{Config.distance_type}_epoch_{str(epoch_idx)}_r1_{recall_1}_r5_{recall_5}_r1050_{recall_10_50}_Loss_{vali_loss}.pt"
        if tag:
            name = name[:-3]+f'_{tag}'+name[-3:]
        if not self.save_dir:
            return
        data = {}
        data['model'] = self.model.state_dict()

        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            data['scheduler'] = self.scheduler.state_dict()

        save_file = os.path.join(self.save_dir, name)
        self.logger.info(f"Saving checkpoint to {save_file}...")
        torch.save(data, save_file)
    
    def load(self, ckp_file=None):
        if ckp_file is None:
            self.logger.info("No checkpoint found!")
            raise FileNotFoundError
        self.logger.info(f"Loading checkpoint from {ckp_file}...")
        checkpoint = torch.load(ckp_file, map_location=torch.device("cpu"))

        self.model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.logger.info(f"Loading optimizer from {ckp_file}...")
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.logger.info(f"Loading scheduler from {ckp_file}...")
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))