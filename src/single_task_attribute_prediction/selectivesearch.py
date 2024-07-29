import torch
import os
import numpy as np
import pandas as pd
from PIL import Image

from selective_search import selective_search

# https://github.com/ChenjieXu/selective_search/tree/master


def read_annotations(path_annotation):
    df = pd.read_csv(path_annotation, delimiter=" ", header=None)
    attr_name = pd.read_csv(
        "./data/VOC2008_attribute/attribute_names.txt", delimiter="\t", header=None
    ).values.tolist()
    class_names = pd.read_csv(
        "./data/VOC2008_attribute/class_names.txt", delimiter="\t", header=None
    ).values.tolist()
    df.columns = ["img", "class", "xmin", "ymin", "xmax", "ymax"] + [
        str for lista in attr_name for str in lista
    ]
    id2category = dict((i + 1, j[0]) for i, j in enumerate(class_names))
    category2id = {v: k for k, v in id2category.items()}
    df["class"] = df["class"].map(category2id)
    images_names = df["img"].unique()
    return df, images_names, class_names, attr_name


def permanent_save(path, list_tensor):
    tensor_dict = {i: tensor for i, tensor in enumerate(list_tensor)}
    torch.save(tensor_dict, path)


def get_selective_search(image):  # image format: H,W,C
    regions = selective_search(image, mode="fast")
    rects = torch.tensor([d for d in regions], dtype=torch.float)
    return rects


def selective_search_images(images_names, path_images, folder_to_saved, filename):
    list_selective_search = []
    os.makedirs(folder_to_saved, exist_ok=True)
    for idx_img, img_path in enumerate(images_names):
        img = Image.open(path_images + img_path)  # W, H
        img = np.asarray(img)  #  (H, W, C)
        rois = get_selective_search(img)  # return torch tensor boxes in xyxy format
        print(f"{idx_img}\t{img_path}\tlen:{rois.shape[0]}")
        list_selective_search.append(rois)
    permanent_save(folder_to_saved + filename, list_selective_search)
    return list_selective_search


if __name__ == "__main__":
    path_annotation = "./data/VOC2008_attribute/ann/ann_train.txt"
    df, images_names, class_names, attr_name = read_annotations(path_annotation)
    list_selective_search = selective_search_images(
        images_names,
        "./data/VOC2008_attribute/train/",
        "./selectivesearch/",
        "list_fast_train_ss_rois.pt",
    )

    path_annotation = "./data/VOC2008_attribute/ann/ann_val.txt"
    df, images_names, class_names, attr_name = read_annotations(path_annotation)
    list_selective_search = selective_search_images(
        images_names,
        "./data/VOC2008_attribute/val/",
        "./selectivesearch/",
        "list_fast_val_ss_rois.pt",
    )
