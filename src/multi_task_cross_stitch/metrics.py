import torch
from config_experiments import config
from bbox_transform import resize_bounding_boxes
import torchmetrics
from bbox_transform import apply_nms
from tqdm import tqdm
import wandb
import logging

def compute_mAP_obj_detect(data_set, model, device):  # train/val

    metric = torchmetrics.detection.MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.5],
        # max_detection_thresholds=[10, 100, 500],  # 20, 60, 100
        extended_summary=True,
    ).to(device)
    metric.warn_on_many_detections = False
    model.eval()

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
    metric.reset()
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


def compute_mAP_attr_predict(data_set, model, device):  # train/val

    metric = torchmetrics.classification.MultilabelAveragePrecision(
        num_labels=config["global"]["num_attributes"], average="none", thresholds=None
    ).to(device)
    metric.warn_on_many_detections = False
    model.eval()

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
            gt_attributes = gt_attributes.to(device)
            ss_rois = ss_rois.to(device)

            indices_batch = data_set.get_indices_batch(
                image.shape[0], gt_bbox.shape[0]
            ).unsqueeze(-1)

            indices_batch = indices_batch.to(device)

            pred_attr, pred_score_attr = model.prediction_rois(
                image,
                gt_bbox,
                indices_batch,
            )

            pred_attr = pred_attr.to(device)
            pred_score_attr = pred_score_attr.to(device)

            preds = pred_score_attr
            target = gt_attributes.int()
            mAP = metric(preds, target)
        mAP = metric.compute()
    metric.reset()
    return mAP


def mAP_view_attributes(mAP, data_test):
    logging.info(mAP)
    logging.info(f"\nmAP@0.50 (per class):")
    for i, value in enumerate(mAP):
        attribute = data_test.id2attribute[i + 1]
        logging.info(f"AP {attribute}: {value:.2f}")
        wandb.config.update({f"AP {attribute} ": value})
    mAP = torch.mean(mAP)

    logging.info(f"\nmAP : {mAP:.2f}")

    wandb.config.update({"mAP_attr@0.50": mAP})
