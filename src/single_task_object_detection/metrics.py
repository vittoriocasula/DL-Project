import torch
from config_experiments import config
from bbox_transform import resize_bounding_boxes
import torchmetrics
from bbox_transform import apply_nms
from tqdm import tqdm
import wandb
import logging
import yaml
import os
from pascal_voc_map import voc_eval


def compute_mAP(data_set, model, device):  # train/val

    metric = torchmetrics.detection.MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.5],
        max_detection_thresholds=[10, 100, 500],
    ).to(device)
    metric.warn_on_many_detections = False
    model.eval()

    """
    with open(
        os.getcwd()
        + '/src/single_task_object_detection/'
        + 'target_mean_std_by_class.yaml',
        'r',
    ) as f:
        mean_std_by_class = yaml.safe_load(f)
    """

    with torch.no_grad():

        for i, (
            image,
            image_size,
            gt_class,
            gt_bbox,
            gt_attributes,
            ss_rois,
        ) in enumerate(tqdm(data_set, desc="Compute mAP")):
            image = image.unsqueeze(0).to(device)
            gt_class = gt_class.to(device)
            gt_bbox = gt_bbox.to(device)
            ss_rois = ss_rois.to(device)

            orig_w, orig_h = image_size
            new_w, new_h = (image.shape[3], image.shape[2])
            gt_bbox = resize_bounding_boxes(
                gt_bbox, orig_size=(new_w, new_h), new_size=(orig_w, orig_h)
            )

            indices_batch = data_set.get_indices_batch(
                image.shape[0], ss_rois.shape[0]
            ).unsqueeze(-1)

            indices_batch = indices_batch.to(device)

            cls_max_score_net, max_score_net, bboxs_net = model.prediction_img(
                image, ss_rois, indices_batch
            )

            bboxs_net = resize_bounding_boxes(
                bboxs_net, orig_size=(new_w, new_h), new_size=(orig_w, orig_h)
            )

            pred_bbox, pred_class, pred_score = apply_nms(
                cls_max_score_net, max_score_net, bboxs_net
            )

            preds = [
                dict(
                    boxes=pred_bbox.to(device),
                    scores=pred_score.to(device),
                    labels=pred_class.to(device),
                )
            ]

            target = [
                dict(
                    boxes=gt_bbox,
                    labels=gt_class,
                )
            ]
            mAP = metric(preds, target)
        mAP = metric.compute()
    return mAP


def view_mAP_for_class(mAP, data_test):
    logging.info(mAP)
    logging.info(f"\nmAP@0.50 (per class):")
    index = torch.arange(1, config["global"]["num_classes"] + 1)

    for i, value in zip(index, mAP["map_per_class"].numpy()):
        category = data_test.id2category.get(i.item())
        mAP_category = value.item()

        logging.info(f"\tAP {category} : {(mAP_category):.2f}")
        wandb.config.update({f"AP {category} ": mAP_category})

    mAP50 = mAP["map_50"].item()
    logging.info(f"\nmAP@0.50 : {mAP50:.2f}")

    wandb.config.update({"mAP@0.50": mAP50})


def compute_pascal_voc_mAP(data_set, model, device):  # train/val

    model.eval()

    """
    with open(
        os.getcwd()
        + '/src/single_task_object_detection/'
        + 'target_mean_std_by_class.yaml',
        'r',
    ) as f:
        mean_std_by_class = yaml.safe_load(f)
    """

    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = [], [], [], [], []
    with torch.no_grad():

        for i, (
            image,
            image_size,
            gt_class,
            gt_bbox,
            gt_attributes,
            ss_rois,
        ) in enumerate(tqdm(data_set, desc="Compute mAP")):
            image = image.unsqueeze(0).to(device)
            gt_class = gt_class.to(device)
            gt_bbox = gt_bbox.to(device)
            ss_rois = ss_rois.to(device)

            orig_w, orig_h = image_size
            new_w, new_h = (image.shape[3], image.shape[2])
            gt_bbox = resize_bounding_boxes(
                gt_bbox, orig_size=(new_w, new_h), new_size=(orig_w, orig_h)
            )

            indices_batch = data_set.get_indices_batch(
                image.shape[0], ss_rois.shape[0]
            ).unsqueeze(-1)

            indices_batch = indices_batch.to(device)

            cls_max_score_net, max_score_net, bboxs_net = model.prediction_img(
                image, ss_rois, indices_batch
            )

            bboxs_net = resize_bounding_boxes(
                bboxs_net, orig_size=(new_w, new_h), new_size=(orig_w, orig_h)
            )

            pred_bbox, pred_class, pred_score = apply_nms(
                cls_max_score_net, max_score_net, bboxs_net
            )

            pred_bboxes.append(pred_bbox)
            pred_labels.append(pred_class)
            pred_scores.append(pred_score)
            gt_bboxes.append(gt_bbox)
            gt_labels.append(gt_class)
    mAP = voc_eval(
        pred_bboxes=pred_bboxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_bboxes=gt_bboxes,
        gt_labels=gt_labels,
        use_07_metric=False,
    )
    return mAP


def view_pascal_voc_mAP_for_class(mAP, data_test):
    logging.info(mAP)
    logging.info(f"\nmAP@0.50 (per class):")
    index = torch.arange(1, config["global"]["num_classes"] + 1)

    for i, value in zip(index, mAP["ap"][1:]):
        category = data_test.id2category.get(i.item())
        mAP_category = value.item()

        logging.info(f"\tAP {category} : {(mAP_category):.2f}")
        wandb.config.update({f"AP {category} ": mAP_category})

    mAP50 = mAP["map"].item()
    logging.info(f"\nmAP@0.50 : {mAP50:.2f}")

    wandb.config.update({"mAP@0.50": mAP50})
