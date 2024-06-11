import torch
import numpy as np
import random


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

