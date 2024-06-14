import torch
from config_experiments import config
from dataloader import get_indices_batch
from bbox_transform import relative_to_absolute_bbox, absolute_to_relative_bbox
import torchmetrics
from bbox_transform import apply_nms
from tqdm import tqdm
import wandb
import logging


def compute_mAP(data_set, model, device):  # train/val

    metric = torchmetrics.detection.MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
        iou_thresholds=[0.5],
        max_detection_thresholds=[10, 100, 500],
    ).to(device)
    metric.warn_on_many_detections = False
    model.eval()

    data_set = tqdm(data_set, desc="Compute mAP")
    with torch.no_grad():

        for i, (
            image,
            image_size,
            gt_class,
            gt_bbox,
            gt_attributes,
            ss_rois,
        ) in enumerate(data_set):
            image = image.unsqueeze(0).to(device)
            gt_class = gt_class.to(device)
            gt_bbox = gt_bbox.to(device)
            ss_rois = ss_rois.to(device)
            gt_bbox = relative_to_absolute_bbox(gt_bbox, image_size)

            _, _, heigth, width = image.shape
            ss_rois = relative_to_absolute_bbox(
                boxes=ss_rois, image_size=(heigth, width)
            )

            indices_batch = get_indices_batch(
                image.shape[0], ss_rois.shape[0]
            ).unsqueeze(-1)

            indices_batch = indices_batch.to(device)

            cls_max_score, max_score, bboxs = model.prediction_img(
                image, ss_rois, indices_batch
            )

            bboxs = absolute_to_relative_bbox(
                bboxs, (heigth, width)
            )
            bboxs = relative_to_absolute_bbox(bboxs, image_size)

            pred_bbox, pred_class, pred_score = apply_nms(
                cls_max_score, max_score, bboxs
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
