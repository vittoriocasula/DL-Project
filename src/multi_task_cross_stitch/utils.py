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


def map_layers_for_init_single_task(backbone):
    if backbone == "alexnet":
        mapping_obj = {
            "cross_stitch_net.models_a.0.0.weight": "alex.features.0.weight",
            "cross_stitch_net.models_a.0.0.bias": "alex.features.0.bias",
            "cross_stitch_net.models_a.1.0.weight": "alex.features.3.weight",
            "cross_stitch_net.models_a.1.0.bias": "alex.features.3.bias",
            "cross_stitch_net.models_a.2.0.weight": "alex.features.6.weight",
            "cross_stitch_net.models_a.2.0.bias": "alex.features.6.bias",
            "cross_stitch_net.models_a.2.2.weight": "alex.features.8.weight",
            "cross_stitch_net.models_a.2.2.bias": "alex.features.8.bias",
            "cross_stitch_net.models_a.2.4.weight": "alex.features.10.weight",
            "cross_stitch_net.models_a.2.4.bias": "alex.features.10.bias",
            "cross_stitch_classifier.branch_a.1.weight": "roi_module.classifier.1.weight",
            "cross_stitch_classifier.branch_a.1.bias": "roi_module.classifier.1.bias",
            "cross_stitch_classifier.branch_a.4.weight": "roi_module.classifier.4.weight",
            "cross_stitch_classifier.branch_a.4.bias": "roi_module.classifier.4.bias",
            "model_obj_detect.cls_score.weight": "obj_detect_head.cls_score.weight",
            "model_obj_detect.cls_score.bias": "obj_detect_head.cls_score.bias",
            "model_obj_detect.bbox.weight": "obj_detect_head.bbox.weight",
            "model_obj_detect.bbox.bias": "obj_detect_head.bbox.bias",
        }

        mapping_attr = {
            "cross_stitch_net.models_b.0.0.weight": "alex.features.0.weight",
            "cross_stitch_net.models_b.0.0.bias": "alex.features.0.bias",
            "cross_stitch_net.models_b.1.0.weight": "alex.features.3.weight",
            "cross_stitch_net.models_b.1.0.bias": "alex.features.3.bias",
            "cross_stitch_net.models_b.2.0.weight": "alex.features.6.weight",
            "cross_stitch_net.models_b.2.0.bias": "alex.features.6.bias",
            "cross_stitch_net.models_b.2.2.weight": "alex.features.8.weight",
            "cross_stitch_net.models_b.2.2.bias": "alex.features.8.bias",
            "cross_stitch_net.models_b.2.4.weight": "alex.features.10.weight",
            "cross_stitch_net.models_b.2.4.bias": "alex.features.10.bias",
            "cross_stitch_classifier.branch_b.1.weight": "roi_module.classifier.1.weight",
            "cross_stitch_classifier.branch_b.1.bias": "roi_module.classifier.1.bias",
            "cross_stitch_classifier.branch_b.4.weight": "roi_module.classifier.4.weight",
            "cross_stitch_classifier.branch_b.4.bias": "roi_module.classifier.4.bias",
            "model_attribute.attr_score.weight": "attribute_head.attr_score.weight",
            "model_attribute.attr_score.bias": "attribute_head.attr_score.bias",
        }

    elif backbone == "vgg16":
        mapping_obj = {
            "cross_stitch_net.models_a.0.0.weight": "alex.features.0.weight",
            "cross_stitch_net.models_a.0.0.bias": "alex.features.0.bias",
            "cross_stitch_net.models_a.0.2.weight": "alex.features.2.weight",
            "cross_stitch_net.models_a.0.2.bias": "alex.features.2.bias",
            "cross_stitch_net.models_a.1.0.weight": "alex.features.5.weight",
            "cross_stitch_net.models_a.1.0.bias": "alex.features.5.bias",
            "cross_stitch_net.models_a.1.2.weight": "alex.features.7.weight",
            "cross_stitch_net.models_a.1.2.bias": "alex.features.7.bias",
            "cross_stitch_net.models_a.2.0.weight": "alex.features.10.weight",
            "cross_stitch_net.models_a.2.0.bias": "alex.features.10.bias",
            "cross_stitch_net.models_a.2.2.weight": "alex.features.12.weight",
            "cross_stitch_net.models_a.2.2.bias": "alex.features.12.bias",
            "cross_stitch_net.models_a.2.4.weight": "alex.features.14.weight",
            "cross_stitch_net.models_a.2.4.bias": "alex.features.14.bias",
            "cross_stitch_net.models_a.3.0.weight": "alex.features.17.weight",
            "cross_stitch_net.models_a.3.0.bias": "alex.features.17.bias",
            "cross_stitch_net.models_a.3.2.weight": "alex.features.19.weight",
            "cross_stitch_net.models_a.3.2.bias": "alex.features.19.bias",
            "cross_stitch_net.models_a.3.4.weight": "alex.features.21.weight",
            "cross_stitch_net.models_a.3.4.bias": "alex.features.21.bias",
            "cross_stitch_net.models_a.4.0.weight": "alex.features.24.weight",
            "cross_stitch_net.models_a.4.0.bias": "alex.features.24.bias",
            "cross_stitch_net.models_a.4.2.weight": "alex.features.26.weight",
            "cross_stitch_net.models_a.4.2.bias": "alex.features.26.bias",
            "cross_stitch_net.models_a.4.4.weight": "alex.features.28.weight",
            "cross_stitch_net.models_a.4.4.bias": "alex.features.28.bias",
            "cross_stitch_classifier.branch_a.1.weight": "roi_module.classifier.1.weight",
            "cross_stitch_classifier.branch_a.1.bias": "roi_module.classifier.1.bias",
            "cross_stitch_classifier.branch_a.4.weight": "roi_module.classifier.4.weight",
            "cross_stitch_classifier.branch_a.4.bias": "roi_module.classifier.4.bias",
            "model_obj_detect.cls_score.weight": "obj_detect_head.cls_score.weight",
            "model_obj_detect.cls_score.bias": "obj_detect_head.cls_score.bias",
            "model_obj_detect.bbox.weight": "obj_detect_head.bbox.weight",
            "model_obj_detect.bbox.bias": "obj_detect_head.bbox.bias",
        }


        mapping_attr = {
            "cross_stitch_net.models_b.0.0.weight": "alex.features.0.weight",
            "cross_stitch_net.models_b.0.0.bias": "alex.features.0.bias",
            "cross_stitch_net.models_b.0.2.weight": "alex.features.2.weight",
            "cross_stitch_net.models_b.0.2.bias": "alex.features.2.bias",
            "cross_stitch_net.models_b.1.0.weight": "alex.features.5.weight",
            "cross_stitch_net.models_b.1.0.bias": "alex.features.5.bias",
            "cross_stitch_net.models_b.1.2.weight": "alex.features.7.weight",
            "cross_stitch_net.models_b.1.2.bias": "alex.features.7.bias",
            "cross_stitch_net.models_b.2.0.weight": "alex.features.10.weight",
            "cross_stitch_net.models_b.2.0.bias": "alex.features.10.bias",
            "cross_stitch_net.models_b.2.2.weight": "alex.features.12.weight",
            "cross_stitch_net.models_b.2.2.bias": "alex.features.12.bias",
            "cross_stitch_net.models_b.2.4.weight": "alex.features.14.weight",
            "cross_stitch_net.models_b.2.4.bias": "alex.features.14.bias",
            "cross_stitch_net.models_b.3.0.weight": "alex.features.17.weight",
            "cross_stitch_net.models_b.3.0.bias": "alex.features.17.bias",
            "cross_stitch_net.models_b.3.2.weight": "alex.features.19.weight",
            "cross_stitch_net.models_b.3.2.bias": "alex.features.19.bias",
            "cross_stitch_net.models_b.3.4.weight": "alex.features.21.weight",
            "cross_stitch_net.models_b.3.4.bias": "alex.features.21.bias",
            "cross_stitch_net.models_b.4.0.weight": "alex.features.24.weight",
            "cross_stitch_net.models_b.4.0.bias": "alex.features.24.bias",
            "cross_stitch_net.models_b.4.2.weight": "alex.features.26.weight",
            "cross_stitch_net.models_b.4.2.bias": "alex.features.26.bias",
            "cross_stitch_net.models_b.4.4.weight": "alex.features.28.weight",
            "cross_stitch_net.models_b.4.4.bias": "alex.features.28.bias",
            "cross_stitch_classifier.branch_b.1.weight": "roi_module.classifier.1.weight",
            "cross_stitch_classifier.branch_b.1.bias": "roi_module.classifier.1.bias",
            "cross_stitch_classifier.branch_b.4.weight": "roi_module.classifier.4.weight",
            "cross_stitch_classifier.branch_b.4.bias": "roi_module.classifier.4.bias",
            "model_attribute.attr_score.weight": "attribute_head.attr_score.weight",
            "model_attribute.attr_score.bias": "attribute_head.attr_score.bias",
        }
    return mapping_obj, mapping_attr
