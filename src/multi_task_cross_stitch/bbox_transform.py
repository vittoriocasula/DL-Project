import torch
import torchvision
from config_experiments import config


def resize_bounding_boxes(bboxes, orig_size, new_size):
    orig_width, orig_height = orig_size
    new_width, new_height = new_size

    # Calculate scale factors
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    # Separate the coordinates
    x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Apply scale factors
    x_min = x_min * scale_w
    x_max = x_max * scale_w
    y_min = y_min * scale_h
    y_max = y_max * scale_h

    # Combine the coordinates back into the tensor
    resized_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    return resized_bboxes


def bbox_offset(proposals, assigned_bb):
    """
    proposals: (N, 4) --> (Px, Py, Pw, Ph)
    assigned_bb: (N,4) --> (Gx, Gy, Gw, Gh)
    """

    proposals = torchvision.ops.box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")
    assigned_bb = torchvision.ops.box_convert(
        assigned_bb, in_fmt="xyxy", out_fmt="cxcywh"
    )

    offset_x = (assigned_bb[:, 0] - proposals[:, 0]) / (proposals[:, 2])
    offset_y = (assigned_bb[:, 1] - proposals[:, 1]) / (proposals[:, 3])
    offset_w = torch.log((assigned_bb[:, 2]) / (proposals[:, 2]))
    offset_h = torch.log((assigned_bb[:, 3]) / (proposals[:, 3]))

    offset = torch.stack([offset_x, offset_y, offset_w, offset_h], dim=1)

    return offset


def regr_to_bbox(proposals, regr, image_size):
    """
    proposals: (N, 4) --> (Px, Py, Pw, Ph)
    regr (N, K, 4) --> (dx, dy, dw, dh) for each class K
    """
    proposals = torchvision.ops.box_convert(proposals, in_fmt="xyxy", out_fmt="cxcywh")

    proposals = proposals.unsqueeze(-1)

    pred_bbox_x = (regr[:, :, 0] * proposals[:, 2, :]) + proposals[:, 0, :]
    pred_bbox_y = (regr[:, :, 1] * proposals[:, 3, :]) + proposals[:, 1, :]
    pred_bbox_w = torch.exp(regr[:, :, 2]) * proposals[:, 2, :]
    pred_bbox_h = torch.exp(regr[:, :, 3]) * proposals[:, 3, :]

    pred_bbox = torch.stack(
        (pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h), dim=-1
    )

    pred_bbox = torchvision.ops.box_convert(pred_bbox, in_fmt="cxcywh", out_fmt="xyxy")
    pred_bbox = torchvision.ops.clip_boxes_to_image(pred_bbox, image_size)
    return pred_bbox


def apply_nms(
    cls_max_score,
    max_score,
    bboxs,
    score_threshold=config["postprocessing"]["score_threshold"],
):
    res_bbox_list = []
    res_cls_list = []
    res_scores_list = []

    num_classes = config["global"]["num_classes"]
    iou_threshold = config["postprocessing"]["nms_iou_threshold"]
    max_roi_per_image = config["postprocessing"]["max_roi_per_image"]

    for c in range(1, num_classes + 1):  # NMS per class independently
        c_mask = cls_max_score == c
        c_bboxs = bboxs[c_mask]
        c_score = max_score[c_mask]

        # Apply score threshold filtering
        score_mask = c_score > score_threshold
        c_bboxs = c_bboxs[score_mask]
        c_score = c_score[score_mask]

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
