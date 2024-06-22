from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torchvision
from bbox_transform import (
    bbox_offset,
    resize_bounding_boxes,
)
from config_experiments import config
import torchvision
import yaml
import os



class VOC08Attr(Dataset):
    def __init__(self, root="./", train=True, transform=None):
        self.train = train
        self.transform = transform
        self.root = root

        if self.train:
            self.df, self.images_names, self.class_names, self.attr_name = (
                self.read_annotations(
                    self.root + "data/VOC2008_attribute/ann/ann_train.txt"
                )
            )
            self.ss_rois = self.permanent_load(
                self.root + "selectivesearch/list_fast_train_ss_rois.pt"
            )
        else:
            self.df, self.images_names, self.class_names, self.attr_name = (
                self.read_annotations(
                    self.root + "data/VOC2008_attribute/ann/ann_val.txt"
                )
            )
            self.ss_rois = self.permanent_load(
                self.root + "selectivesearch/list_fast_test_ss_rois.pt"
            )

        self.id2category = dict((i + 1, j[0]) for i, j in enumerate(self.class_names))
        self.category2id = {v: k for k, v in self.id2category.items()}
        self.id2attribute = dict((i + 1, j[0]) for i, j in enumerate(self.attr_name))
        self.attribute2id = {v: k for k, v in self.id2attribute.items()}


        with open(
            os.getcwd()
            + "/src/single_task_object_detection/"
            + "target_mean_std_by_class.yaml",
            "r",
        ) as f:
            self.mean_std_by_class = yaml.safe_load(f)

    def __len__(self):
        return len(self.images_names)

    def read_annotations(self, path_annotation):
        df = pd.read_csv(path_annotation, delimiter=" ", header=None)
        attr_name = pd.read_csv(
            self.root + "data/VOC2008_attribute/attribute_names.txt",
            delimiter="\t",
            header=None,
        ).values.tolist()
        class_names = pd.read_csv(
            self.root + "data/VOC2008_attribute/class_names.txt",
            delimiter="\t",
            header=None,
        ).values.tolist()
        df.columns = ["img", "class", "xmin", "ymin", "xmax", "ymax"] + [
            str for lista in attr_name for str in lista
        ]
        id2category = dict((i + 1, j[0]) for i, j in enumerate(class_names))
        category2id = {v: k for k, v in id2category.items()}
        df["class"] = df["class"].map(category2id)
        images_names = df["img"].unique()
        return df, images_names, class_names, attr_name

    def extract_annotations_image(self, df, filename):
        df_label = df.loc[df["img"] == filename]
        gt_bbox = torch.tensor(
            df_label[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float
        )
        gt_category = torch.tensor(
            df_label[["class"]].values, dtype=torch.long
        ).flatten()
        gt_attributes = torch.tensor(df_label.iloc[:, 6:].values, dtype=torch.float)
        return (gt_category, gt_bbox, gt_attributes)

    def permanent_load(self, path):
        list_sample_roi = torch.load(path)
        loaded_tensor_list = [list_sample_roi[key] for key in list_sample_roi.keys()]
        return loaded_tensor_list

    def random_horizontal_flip_with_boxes(self, image, gt_bbox, ss_rois, p=0.5):
        if torch.rand(1) < p:
            image = torchvision.transforms.functional.hflip(image)
            image_width = image.shape[2]
            gt_bbox = self.flip_boxes(gt_bbox, image_width)
            ss_rois = self.flip_boxes(ss_rois, image_width)

        return image, gt_bbox, ss_rois

    def flip_boxes(self, boxes, image_width):
        flipped_boxes = boxes.clone()
        flipped_boxes[:, 0] = image_width - boxes[:, 2]
        flipped_boxes[:, 2] = image_width - boxes[:, 0]
        return flipped_boxes

    def __getitem__(self, idx):
        img_path = self.images_names[idx]
        if self.train:
            image = Image.open(self.root + "data/VOC2008_attribute/train/" + img_path)
        else:
            image = Image.open(self.root + "data/VOC2008_attribute/val/" + img_path)
        img_size_orig = image.size  # W, H (original)
        gt_class, gt_bbox, gt_attributes = self.extract_annotations_image(
            self.df, img_path
        )
        ss_rois = self.ss_rois[idx]
        if self.transform:
            image = self.transform(image)


            orig_w, orig_h = img_size_orig
            new_w, new_h = (image.shape[2], image.shape[1])

            gt_bbox = resize_bounding_boxes(
                gt_bbox, orig_size=(orig_w, orig_h), new_size=(new_w, new_h)
            )
            ss_rois = resize_bounding_boxes(
                ss_rois, orig_size=(orig_w, orig_h), new_size=(new_w, new_h)
            )

            if self.train:
                image, gt_bbox, ss_rois = self.random_horizontal_flip_with_boxes(
                    image, gt_bbox, ss_rois
                )
        return image, img_size_orig, gt_class, gt_bbox, gt_attributes, ss_rois

    def extract_positive_and_negative(self, gt_class, gt_bbox, gt_attributes, ss_rois):
        train_roi = []
        train_cls = []
        train_offset = []
        train_attr = []

        ious = torchvision.ops.box_iou(ss_rois, gt_bbox)
        max_ious = ious.max(axis=1).values
        max_idx = ious.argmax(axis=1)
        offset_bbox = bbox_offset(ss_rois, gt_bbox[max_idx])

        for j in range(len(ss_rois)):  # j indica la specifica ROI (SS)
            if max_ious[j] < config["preprocessing"]["iou_thresh_low_neg"]:
                continue
            train_roi.append(ss_rois[j])
            train_offset.append(offset_bbox[j])

            if max_ious[j] >= config["preprocessing"]["iou_thresh_low_pos"]:
                train_cls.append(
                    gt_class[max_idx[j].unsqueeze(0)]
                )  # gt_bbox associata alla roi
                train_attr.append(gt_attributes[max_idx[j]])
            else:
                train_cls.append(torch.tensor([0]))  # background
                train_attr.append(torch.zeros(config["global"]["num_attributes"]))

        train_roi = torch.stack(train_roi)
        train_cls = torch.cat(train_cls)
        train_offset = torch.stack(train_offset)
        train_attr = torch.stack(train_attr)

        return train_roi, train_cls, train_offset, train_attr

    def collate_fn(self, batch):
        images, img_size_original, gt_class, gt_bbox, gt_attributes, ss_rois = zip(
            *batch
        )  # img_size sono originali= (W,H)

        image_shapes = [
            img.shape for img in images
        ]  # considero le dimensioni dopo aver effettuato il resize per mantenere l'aspect ratio
        max_height = max(shape[1] for shape in image_shapes)  # max heigth
        max_width = max(shape[2] for shape in image_shapes)  # max width

        batch_images = torch.zeros(
            size=(len(images), images[0].shape[0], max_height, max_width)
        )
        for i, img in enumerate(images):
            _, height, width = img.shape
            batch_images[i, :, :height, :width] = img

        rois = []
        classes = []
        offsets = []
        attrs = []
        indices_batch = []
        for i in range(len(batch)):

            train_roi, train_cls, train_offset, train_attr = (
                self.extract_positive_and_negative(
                    gt_class[i], gt_bbox[i], gt_attributes[i], ss_rois[i]
                )
            )

            # Normalizzazione del train_offset
            normalized_train_offset = torch.empty_like(train_offset)
            for id, cls in enumerate(train_cls):
                if cls.item() != 0:
                    mean = torch.tensor(self.mean_std_by_class[str(cls.item())]["mean"])
                    std = torch.tensor(self.mean_std_by_class[str(cls.item())]["std"])
                    normalized_train_offset[id] = (train_offset[id] - mean) / std
                else:
                    normalized_train_offset[id] = train_offset[id]


            idx_batch = torch.full(size=(train_cls.shape[0],), fill_value=i)
            # ottieni tutti i positivi e i negativi

            rois.append(train_roi)
            classes.append(train_cls)
            offsets.append(normalized_train_offset) 
            attrs.append(train_attr)
            indices_batch.append(idx_batch)

        # prendi i positivi e negativi nella forma 1:3 nel batch

        rois = torch.cat(rois)
        classes = torch.cat(classes)
        offsets = torch.cat(offsets)
        attrs = torch.cat(attrs)
        indices_batch = torch.cat(indices_batch)

        pos_indices = classes.nonzero().squeeze(-1)  # tensor of indices
        neg_indices = (
            (classes == 0).nonzero(as_tuple=False).squeeze(-1)
        )  # tensor of indices

        n_pos = int(
            config["preprocessing"]["n_images"]
            * config["preprocessing"]["n_roi_per_image"]
            * config["preprocessing"]["ratio_pos_roi"]
        )  # positive roi
        n_neg = (
            config["preprocessing"]["n_images"]
            * config["preprocessing"]["n_roi_per_image"]
            - n_pos
        )  # negative roi

        pos_selected = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        neg_selected = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        selected = torch.cat([pos_selected, neg_selected])

        rois = rois[selected]
        classes = classes[selected]
        offsets = offsets[selected]
        attrs = attrs[selected]
        indices_batch = indices_batch[selected].unsqueeze(-1)

        return batch_images, rois, classes, offsets, attrs, indices_batch

    def get_indices_batch(self, n_images, n_roi_per_image):
        batches = []
        for i in range(n_images):
            current_batch = torch.full((n_roi_per_image,), fill_value=i)
            batches.append(current_batch)
        indices_batch = torch.cat(batches)
        return indices_batch
