import torch
import numpy as np
import random
import logging

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(idx_gpu=0):
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(idx_gpu))
    else:
        device = torch.device("cpu")
    return device

def log_dict(d, parent_key=''):
    for key, value in d.items():
        if isinstance(value, dict):
            log_dict(value, f'{parent_key}{key}.')
        else:
            logging.info(f'{parent_key}{key}: {value}')