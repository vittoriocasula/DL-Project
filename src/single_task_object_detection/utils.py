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


def get_indices_batch(n_images, n_roi_per_image):
    batches = []
    for i in range(n_images):
        current_batch = torch.full((n_roi_per_image,), fill_value=i)
        batches.append(current_batch)
    indices_batch = torch.cat(batches)
    return indices_batch
"""

def get_indices_batch_by_list(n_images, list_n_roi):
    batches = []
    for i in range(n_images):
        current_batch = torch.full((list_n_roi[i],), fill_value=i)
        batches.append(current_batch)
    indices_batch = torch.cat(batches)
    return indices_batch
"""

