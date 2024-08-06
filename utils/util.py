import logging
import os
import sys

import random
import numpy
import torch
import torch.backends.cudnn

from config import Config

def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(name, log_dir=None):
    # =================================================================
    # 创建logger记录器
    # =================================================================
    logger = logging.getLogger(name)
    # 设置记录器记录级别
    logger.setLevel(logging.DEBUG)
    logger.propagate=False

    # =================================================================
    # 创建指向屏幕输出流的分发处理器Handler
    # =================================================================
    # stream_handler = logging.StreamHandler(stream=sys.stdout)
    # stream_handler.setLevel(logging.DEBUG)

    # =================================================================
    #  创建格式化器formatter 
    # =================================================================
    formatter = logging.Formatter("[%(asctime)s] - [%(filename)s line:%(lineno)d] %(levelname)s => %(message)s", datefmt="%m-%d %H:%M")

    # 将formatter添加到stream_handler处理器
    # stream_handler.setFormatter(formatter)
    # 将stream_handler处理器添加到logger记录器
    # logger.addHandler(stream_handler)  # 需要在控制台输出则取消注释

    # =================================================================
    # 创建指向文件的分发处理器Handler 
    # =================================================================
    if log_dir is not None:
        log_path = os.path.join(log_dir, 'train.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        # 将formatter添加到file_handler处理器
        file_handler.setFormatter(formatter)
        # 将file_handler处理器添加到logger记录器
        logger.addHandler(file_handler)

    return logger