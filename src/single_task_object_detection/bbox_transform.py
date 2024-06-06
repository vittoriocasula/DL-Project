import torch
import torchvision
from config_experiments import config


def absolute_to_relative_bbox(boxes, image_size):
    """
    Convert bounding boxes from absolute coordinates to relative coordinates.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in format (x1, y1, x2, y2).
        image_size (tuple): Tuple containing (width, height) of the image.

    Returns:
        Tensor: Tensor of shape (N, 4) containing bounding boxes in relative coordinates.
    """
    width, height = image_size
    boxes = boxes.clone()
    boxes[:, 0] /= width
    boxes[:, 1] /= height
    boxes[:, 2] /= width
    boxes[:, 3] /= height
    return boxes


def relative_to_absolute_bbox(boxes, image_size):
    """
    Convert bounding boxes from relative coordinates to absolute coordinates.

    Args:
        boxes (Tensor): Tensor of shape (N, 4) containing bounding boxes in format (x1, y1, x2, y2).
        image_size (tuple): Tuple containing (width, height) of the image.

    Returns:
        Tensor: Tensor of shape (N, 4) containing bounding boxes in absolute coordinates.
    """
    width, height = image_size
    boxes = boxes.clone()
    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height
    return boxes


def bbox_offset(proposals, assigned_bb, eps=1e-6):
    """
    proposals: (N, 4) --> (Px, Py, Pw, Ph)
    assigned_bb: (N,4) --> (Gx, Gy, Gw, Gh)
    """
    proposals = torchvision.ops.box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")
    assigned_bb = torchvision.ops.box_convert(
        assigned_bb, in_fmt="xyxy", out_fmt="cxcywh"
    )

    offset_x = (assigned_bb[:, 0] - proposals[:, 0]) / (proposals[:, 2] + eps)
    offset_y = (assigned_bb[:, 1] - proposals[:, 1]) / (proposals[:, 3] + eps)
    offset_w = torch.log((assigned_bb[:, 2] + eps) / (proposals[:, 2] + eps))
    offset_h = torch.log((assigned_bb[:, 3] + eps) / (proposals[:, 3] + eps))

    offset = torch.stack([offset_x, offset_y, offset_w, offset_h], dim=1)
    offset_normalized = (
        offset - torch.tensor(config["preprocessing"]["bbox_normalize_means"])
    ) / torch.tensor(config["preprocessing"]["bbox_normalize_stds"])

    return offset_normalized


def regr_to_bbox(proposals, regr, image_size):
    """
    proposals: (N, 4) --> (Px, Py, Pw, Ph)
    offset_preds (N, K, 4) --> (dx, dy, dw, dh) for each class K
    """

    proposals = torchvision.ops.box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")
    proposals = proposals.unsqueeze(-1)

    un_normalized_regr_x = (
        (regr[:, :, 0]) * config["preprocessing"]["bbox_normalize_stds"][0]
    ) + config["preprocessing"]["bbox_normalize_means"][0]

    un_normalized_regr_y = (
        (regr[:, :, 1]) * config["preprocessing"]["bbox_normalize_stds"][1]
    ) + config["preprocessing"]["bbox_normalize_means"][1]

    un_normalized_regr_w = (
        (regr[:, :, 2]) * config["preprocessing"]["bbox_normalize_stds"][2]
    ) + config["preprocessing"]["bbox_normalize_means"][2]

    un_normalized_regr_h = (
        (regr[:, :, 3]) * config["preprocessing"]["bbox_normalize_stds"][3]
    ) + config["preprocessing"]["bbox_normalize_means"][3]

    pred_bbox_x = (un_normalized_regr_x * proposals[:, 2, :]) + proposals[:, 0, :]
    pred_bbox_y = (un_normalized_regr_y * proposals[:, 3, :]) + proposals[:, 1, :]
    pred_bbox_w = torch.exp(un_normalized_regr_w) * proposals[:, 2, :]
    pred_bbox_h = torch.exp(un_normalized_regr_h) * proposals[:, 3, :]

    pred_bbox = torch.stack((pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h), dim=2)

    pred_bbox = torchvision.ops.box_convert(pred_bbox, in_fmt="cxcywh", out_fmt="xyxy")
    pred_bbox = torchvision.ops.clip_boxes_to_image(pred_bbox, image_size)
    return pred_bbox


"""
def apply_nms(cls_max_score, max_score, bboxs):

    res_bbox = []
    res_cls = []
    res_scores = []
    for c in range(
        1, config["global"]["num_classes"] + 1
    ):  # nms per classe indipendente
        c_mask = cls_max_score == c
        c_bboxs = bboxs[c_mask]
        c_score = max_score[c_mask]

        if len(c_bboxs) > 0:
            nms_idxs = torchvision.ops.nms(
                c_bboxs,
                c_score,
                iou_threshold=config["postprocessing"]["iou_threshold"],
            )
            # Limiting to top 100 detections per class
            nms_idxs = nms_idxs[: config["postprocessing"]["max_roi_per_image"]]

            res_bbox.extend(c_bboxs[nms_idxs])
            res_cls.extend(torch.tensor([c] * len(nms_idxs)))
            res_scores.extend(c_score[nms_idxs])

    if res_bbox:
        res_bbox = torch.cat([bbox.view(-1, 4) for bbox in res_bbox], dim=0)
        res_cls = torch.cat([classe.view(1) for classe in res_cls], dim=0)
        res_scores = torch.cat([score.view(1) for score in res_scores], dim=0)

    else:  # res_bbox empty list
        res_bbox = torch.tensor(res_bbox)
        res_cls = torch.tensor(res_cls)
        res_scores = torch.tensor(res_scores)
    return res_bbox, res_cls, res_scores
"""


def apply_nms(cls_max_score, max_score, bboxs):

    res_bbox_list = []
    res_cls_list = []
    res_scores_list = []

    num_classes = config["global"]["num_classes"]
    iou_threshold = config["postprocessing"]["iou_threshold"]
    max_roi_per_image = config["postprocessing"]["max_roi_per_image"]

    for c in range(1, num_classes + 1):  # nms per classe indipendente
        c_mask = cls_max_score == c
        c_bboxs = bboxs[c_mask]
        c_score = max_score[c_mask]

        if len(c_bboxs) > 0:
            nms_idxs = torchvision.ops.nms(c_bboxs, c_score, iou_threshold)
            nms_idxs = nms_idxs[:max_roi_per_image]

            res_bbox_list.append(c_bboxs[nms_idxs])
            res_cls_list.append(torch.full((len(nms_idxs),), c, dtype=torch.int64))
            res_scores_list.append(c_score[nms_idxs])

    if res_bbox_list:
        res_bbox = torch.cat(res_bbox_list, dim=0)
        res_cls = torch.cat(res_cls_list, dim=0)
        res_scores = torch.cat(res_scores_list, dim=0)
    else:
        res_bbox = torch.empty((0, 4), dtype=bboxs.dtype, device=bboxs.device)
        res_cls = torch.empty((0,), dtype=torch.int64, device=bboxs.device)
        res_scores = torch.empty((0,), dtype=max_score.dtype, device=max_score.device)

    return res_bbox, res_cls, res_scores
