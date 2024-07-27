import sys
import cv2
import torch
import os

def get_file_names(directory):
    file_names = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            file_names.append(entry)
    return file_names

def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


def config(gs, img, strategy="f"):
    gs.setBaseImage(img)

    if strategy == "s":
        gs.switchToSingleStrategy()
    elif strategy == "f":
        gs.switchToSelectiveSearchFast()
    elif strategy == "q":
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects

def apply_ss_to_img(path_img):
    gs = get_selective_search()
    img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    config(gs, img, strategy="f")
    rects = get_rects(gs)
    return rects

def permanent_save(path, list_tensor):
    tensor_dict = {i: tensor for i, tensor in enumerate(list_tensor)}
    torch.save(tensor_dict, path)


if __name__ == "__main__":
    folder_to_saved = "selective_search/"
    filename = "list_cv2_ss_rois_fast_train.pt"
    os.makedirs(folder_to_saved, exist_ok=True)
    train_directory = "./data/VOC2008_attribute/train/"
    images_names = get_file_names(train_directory)
    list_selective_search = []

    for idx_img, path_img in enumerate(images_names):
        ss_rois = torch.tensor(apply_ss_to_img(train_directory + path_img), dtype = torch.float32)
        print(f"{idx_img}\t{path_img}\tlen:{ss_rois.shape[0]}")
        list_selective_search.append(ss_rois)
    permanent_save(folder_to_saved + filename, list_selective_search)


    folder_to_saved = "selective_search/"
    filename = "list_cv2_ss_rois_fast_test.pt"
    os.makedirs(folder_to_saved, exist_ok=True)
    val_directory = "./data/VOC2008_attribute/val/"
    images_names = get_file_names(val_directory)
    list_selective_search = []

    for idx_img, path_img in enumerate(images_names):
        ss_rois = torch.tensor(apply_ss_to_img(val_directory + path_img), dtype = torch.float32)
        print(f"{idx_img}\t{path_img}\tlen:{ss_rois.shape[0]}")
        list_selective_search.append(ss_rois)
    permanent_save(folder_to_saved + filename, list_selective_search)
