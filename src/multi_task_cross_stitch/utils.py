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


def log_dict(d, parent_key=""):
    for key, value in d.items():
        if isinstance(value, dict):
            log_dict(value, f"{parent_key}{key}.")
        else:
            logging.info(f"{parent_key}{key}: {value}")


def get_map_step(epoch, max_epoch, steps):
    progress = epoch / max_epoch
    i = 0
    while i < (len(steps) - 1) and progress > steps[i]["progress"]:
        i += 1
    return steps[i]["step"]


def copy_weights(src_layers, dst_layers):
    src_idx = 0
    dst_idx = 0
    while src_idx < len(src_layers) and dst_idx < len(dst_layers):
        if isinstance(dst_layers[dst_idx], torch.nn.Conv2d):
            if isinstance(src_layers[src_idx], torch.nn.Conv2d):
                dst_layers[dst_idx].weight.data = src_layers[
                    src_idx
                ].weight.data.clone()
                if src_layers[src_idx].bias is not None:
                    dst_layers[dst_idx].bias.data = src_layers[
                        src_idx
                    ].bias.data.clone()
            src_idx += 1
        dst_idx += 1