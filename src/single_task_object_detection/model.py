import torchvision
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights
from config_experiments import config
from bbox_transform import regr_to_bbox


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.rawnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.rawnet.features.children())[:-1])

    def forward(self, input):
        return self.features(input)


class ROI_Module(nn.Module):
    def __init__(self):
        super(ROI_Module, self).__init__()

        self.roipool = torchvision.ops.RoIPool(
            output_size=(6, 6), spatial_scale=config["model"]["spatial_scale"]
        )

        self.feature = nn.Sequential(
            *[
                layer
                for layer in list(
                    torchvision.models.alexnet(
                        weights=AlexNet_Weights.DEFAULT
                    ).classifier.children()
                )[:-1]
            ]
        )

    def forward(self, features, rois, ridx):
        idx_rois = torch.cat(
            (ridx, rois), dim=-1
        )  # create matrix with (batch_idx, rois(xyxy))
        res = self.roipool(features, idx_rois)
        res = res.view(res.size(0), -1)
        feat = self.feature(res)
        return feat


class ObjectDetectionHead(nn.Module):
    def __init__(self, num_classes=config["global"]["num_classes"]):
        super().__init__()
        self.num_classes = num_classes

        self.cls_score = nn.Linear(4096, self.num_classes + 1)
        self.bbox = nn.Linear(4096, 4 * (self.num_classes + 1))

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.normal_(self.bbox.weight, mean=0, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox.bias, 0)

    def forward(self, feat):
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, self.num_classes + 1, 4)
        return cls_score, bbox


class ObjectDetectionModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.alex = Backbone()
        self.roi_module = ROI_Module()
        self.obj_detect_head = ObjectDetectionHead(
            num_classes=config["global"]["num_classes"]
        )

    def forward(self, x, rois, ridx):
        out = self.alex(x)
        out = self.roi_module(out, rois, ridx)
        cls_score, bbox = self.obj_detect_head(out)
        return cls_score, bbox

    def prediction_img(self, img, rois, ridx):
        self.eval()
        score, tbbox = self(img, rois, ridx)
        score = nn.functional.softmax(score, dim=-1)
        max_score, cls_max_score = torch.max(score, dim=-1)
        bboxs = regr_to_bbox(
            rois, tbbox, config["transform"]["resize_values"]
        )  # bbox relative
        classes = cls_max_score.view(-1, 1, 1).expand(cls_max_score.size(0), 1, 4)
        bboxs = bboxs.gather(1, classes).squeeze(1)
        return cls_max_score, max_score, bboxs

    def calc_loss(
        self,
        probs,
        bbox,  # offset calcolati in base alle roi assolute
        labels,
        gt_bbox,  # offset gt relativi
    ):
        cel = nn.CrossEntropyLoss()
        sl1 = nn.SmoothL1Loss()
        loss_sc = cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        # ignore background
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)

        loss_sc = config["loss"]["lmb_cls"] * loss_sc
        loss_loc = config["loss"]["lmb_loc"] * loss_loc
        loss = loss_sc + loss_loc
        return loss, loss_sc, loss_loc