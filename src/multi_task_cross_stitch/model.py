import torchvision
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights, VGG16_Weights
from config_experiments import config
from bbox_transform import regr_to_bbox
import numpy as np
import wandb


class Backbone(nn.Module):
    def __init__(self, model_type=config["model"]["backbone"]):
        super(Backbone, self).__init__()

        if model_type == "alexnet":
            rawnet = torchvision.models.alexnet(
                weights=torchvision.models.AlexNet_Weights.DEFAULT
            )
        elif model_type == "vgg16":
            rawnet = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.DEFAULT
            )
        else:
            raise ValueError(
                "Model type not supported. Choose between 'alexnet' and 'vgg16'."
            )

        self.features = nn.Sequential(*list(rawnet.features.children())[:-1])

        # Freeza il primo layer conv
        self.features[0].weight.requires_grad = False
        self.features[0].bias.requires_grad = False

    def forward(self, input):
        return self.features(input)


class ROI_Module(nn.Module):
    def __init__(self):
        super(ROI_Module, self).__init__()

        self.roipool = torchvision.ops.RoIPool(
            output_size=(
                config["model"]["output_size_roipool"][0],
                config["model"]["output_size_roipool"][1],
            ),
            spatial_scale=config["model"]["spatial_scale"],
        )

    def forward(self, features, rois, ridx):
        idx_rois = torch.cat(
            (ridx, rois), dim=-1
        )  # create matrix with (batch_idx, rois(xyxy))
        res = self.roipool(features, idx_rois)
        res = res.view(res.size(0), -1)
        return res


class CrossStitchROI_Module(nn.Module):
    def __init__(self, roi_a, roi_b):
        super(CrossStitchROI_Module, self).__init__()
        self.roi_a = roi_a
        self.roi_b = roi_b
        self.cs = CrossStitchUnit()

    def forward(self, x_a, x_b, rois, ridx):
        x_a = self.roi_a(x_a, rois, ridx)
        x_b = self.roi_b(x_b, rois, ridx)
        x_a, x_b = self.cs(x_a, x_b)
        return x_a, x_b


class CrossStitchClassifier(nn.Module):
    def __init__(self):
        super(CrossStitchClassifier, self).__init__()

        self.branch_a = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                config["model"]["n_unit_classifier"]
                * config["model"]["output_size_roipool"][0]
                * config["model"]["output_size_roipool"][1],
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.branch_b = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                config["model"]["n_unit_classifier"]
                * config["model"]["output_size_roipool"][0]
                * config["model"]["output_size_roipool"][1],
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.cross_stitch_1 = CrossStitchUnit()
        self.cross_stitch_2 = CrossStitchUnit()

    def forward(self, x_a, x_b):

        x_a = self.branch_a[0:4](x_a)
        x_b = self.branch_b[0:4](x_b)
        x_a, x_b = self.cross_stitch_1(x_a, x_b)
        x_a = self.branch_a[4:](x_a)
        x_b = self.branch_b[4:](x_b)
        x_a, x_b = self.cross_stitch_2(x_a, x_b)
        return x_a, x_b


class AttributePredictionHead(nn.Module):
    def __init__(self, num_attributes=config["global"]["num_attributes"]):
        super().__init__()
        self.num_attributes = num_attributes

        self.attr_score = nn.Linear(4096, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, feat):
        attr_score = self.attr_score(feat)
        return attr_score


class AttributePredictionModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = Backbone()
        self.roi_module = ROI_Module()
        self.attribute_head = AttributePredictionHead(
            num_attributes=config["global"]["num_attributes"]
        )

    def forward(self, x, rois, ridx):
        out = self.backbone(x)
        out = self.roi_module(out, rois, ridx)
        score_attr = self.attribute_head(out)
        return score_attr

    def prediction_rois(self, img, rois, ridx):
        self.eval()
        with torch.no_grad():
            sc_attr = self(img, rois, ridx)
        score_attr = nn.functional.sigmoid(sc_attr)
        attr = torch.where(sc_attr > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        return attr, score_attr

    def calc_loss(
        self,
        attr,
        labels,
        gt_attr,
    ):

        bce = nn.BCEWithLogitsLoss()

        if (
            len(attr[labels != 0]) == 0 or len(gt_attr[labels != 0]) == 0
        ):  # gestisci il caso in cui non c'è bbox positivi
            loss_attr = torch.tensor(0.0, requires_grad=True).to(labels.device)
        else:
            loss_attr = bce(attr[labels != 0], gt_attr[labels != 0])
        return loss_attr


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
        self.backbone = Backbone()
        self.roi_module = ROI_Module()
        self.obj_detect_head = ObjectDetectionHead(
            num_classes=config["global"]["num_classes"]
        )

    def forward(self, x, rois, ridx):
        out = self.backbone(x)
        out = self.roi_module(out, rois, ridx)
        cls_score, bbox = self.obj_detect_head(out)
        return cls_score, bbox

    def prediction_img(self, img, rois, ridx):
        self.eval()
        with torch.no_grad():
            score, tbbox = self(img, rois, ridx)
        _, _, heigth, width = img.shape
        score = nn.functional.softmax(score, dim=-1)
        max_score, cls_max_score = torch.max(score, dim=-1)

        bboxs = regr_to_bbox(rois, tbbox, (heigth, width))

        bboxs = bboxs[torch.arange(cls_max_score.shape[0]), cls_max_score]
        return cls_max_score, max_score, bboxs

    def calc_loss(
        self,
        probs,
        bbox,
        labels,
        gt_bbox,
    ):

        cel = nn.CrossEntropyLoss()

        sl1 = nn.SmoothL1Loss(reduction="none")
        loss_sc = cel(probs, labels)

        mask = (labels != 0).bool()
        t_u = bbox[torch.arange(bbox.shape[0]), labels]
        loss_loc = (
            torch.sum(torch.sum(sl1(t_u[mask], gt_bbox[mask]), dim=1), dim=0)
            / labels.shape[0]
        )

        loss_sc = config["loss"]["lmb_cls"] * loss_sc
        loss_loc = config["loss"]["lmb_loc"] * loss_loc
        loss = loss_sc + loss_loc
        return loss, loss_sc, loss_loc


class CrossStitchBackbone(nn.Module):
    def __init__(self, backbone_a, backbone_b, model_type=config["model"]["backbone"]):
        super().__init__()

        if model_type == "alexnet":
            cs_position = [(0, 3), (3, 6), (6, 12)]

        elif model_type == "vgg16":
            cs_position = [(0, 5), (5, 10), (10, 17), (17, 24), (24, 30)]

        self.models_a = nn.ModuleList(
            [nn.Sequential(*backbone_a[i:j]) for i, j in cs_position]
        )
        self.models_b = nn.ModuleList(
            [nn.Sequential(*backbone_b[i:j]) for i, j in cs_position]
        )

        # Create CrossStitchUnits for each pooling layer
        self.cross_stitch_units = nn.ModuleList(
            [CrossStitchUnit() for _ in range(len(cs_position) - 1)]
        )

    def forward(self, x):
        out_a, out_b = x, x

        # Apply all but the last cross-stitch unit
        for model_a, model_b, cross_stitch in zip(
            self.models_a[:-1], self.models_b[:-1], self.cross_stitch_units
        ):
            out_a, out_b = model_a(out_a), model_b(out_b)
            out_a, out_b = cross_stitch(out_a, out_b)

        # Apply the last part of the models without a cross-stitch unit
        out_a, out_b = self.models_a[-1](out_a), self.models_b[-1](out_b)

        return out_a, out_b


class CrossStitchUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.alfa_a = nn.Parameter(
            torch.tensor(config["cross_stitch"]["alfa_a_init"], dtype=torch.float32),
            requires_grad=True,
        )
        self.alfa_b = nn.Parameter(
            torch.tensor(config["cross_stitch"]["alfa_b_init"], dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, xa, xb):
        new_xa = self.alfa_a[0] * xa + self.alfa_a[1] * xb
        new_xb = self.alfa_b[0] * xa + self.alfa_b[1] * xb
        return new_xa, new_xb

    def log_parameters(self):
        # Loggare i valori di alfa_a e alfa_b su wandb
        wandb.log(
            {
                "alfa_a_0": self.alfa_a[0].item(),
                "alfa_a_1": self.alfa_a[1].item(),
                "alfa_b_0": self.alfa_b[0].item(),
                "alfa_b_1": self.alfa_b[1].item(),
            }
        )


class CrossStitchNet(nn.Module):
    def __init__(self, backbone_a, backbone_b):
        super().__init__()
        self.cross_stitch_net = CrossStitchBackbone(backbone_a.features, backbone_b.features)
        self.roi_a = ROI_Module()
        self.roi_b = ROI_Module()

        self.cs_roi = CrossStitchROI_Module(self.roi_a, self.roi_b)

        self.cross_stitch_classifier = CrossStitchClassifier()

        self.model_obj_detect = ObjectDetectionHead(
            num_classes=config["global"]["num_classes"]
        )
        self.model_attribute = AttributePredictionHead(
            num_attributes=config["global"]["num_attributes"]
        )

    def forward(self, x, rois, ridx):
        out_a, out_b = self.cross_stitch_net(x)
        out_a, out_b = self.cs_roi(out_a, out_b, rois, ridx)
        out_a, out_b = self.cross_stitch_classifier(out_a, out_b)
        cls_score, bbox = self.model_obj_detect(out_a)
        attr_score = self.model_attribute(out_b)
        return cls_score, bbox, attr_score  # cls_score, bbox, attr_score

    def prediction_img(self, img, rois, ridx):
        self.eval()
        with torch.no_grad():
            score, tbbox, _ = self(img, rois, ridx)
        _, _, heigth, width = img.shape
        score = nn.functional.softmax(score, dim=-1)
        max_score, cls_max_score = torch.max(score, dim=-1)

        bboxs = regr_to_bbox(rois, tbbox, (heigth, width))

        bboxs = bboxs[torch.arange(cls_max_score.shape[0]), cls_max_score]
        # bboxs = self.unnormalize_bbox(bboxs)
        return cls_max_score, max_score, bboxs

    def prediction_rois(self, img, rois, ridx):
        self.eval()
        with torch.no_grad():
            _, _, sc_attr = self(img, rois, ridx)
        score_attr = nn.functional.sigmoid(sc_attr)
        attr = torch.where(
            score_attr > 0.5,
            torch.tensor(1.0, device=sc_attr.device),
            torch.tensor(0.0, device=sc_attr.device),
        )
        return attr, score_attr

    def calc_loss(
        self,
        probs,
        bbox,
        labels,
        gt_bbox,
        attr,
        gt_attr,
    ):

        cel = nn.CrossEntropyLoss()
        sl1 = nn.SmoothL1Loss(reduction="none")
        bce = nn.BCEWithLogitsLoss()

        loss_sc = cel(probs, labels)

        mask = (labels != 0).bool()
        t_u = bbox[torch.arange(bbox.shape[0]), labels]
        loss_loc = (
            torch.sum(torch.sum(sl1(t_u[mask], gt_bbox[mask]), dim=1), dim=0)
            / labels.shape[0]
        )
        if (
            len(attr[labels != 0]) == 0 or len(gt_attr[labels != 0]) == 0
        ):  # gestiscri il caso in cui non c'è bbox positivi
            loss_attr = torch.tensor(0.0, requires_grad=True).to(labels.device)
        else:
            loss_attr = bce(attr[labels != 0], gt_attr[labels != 0])

        loss_sc = config["loss"]["lmb_cls"] * loss_sc
        loss_loc = config["loss"]["lmb_loc"] * loss_loc
        loss_attr = config["loss"]["lmb_attr"] * loss_attr

        loss = loss_sc + loss_loc

        loss = loss_sc + loss_loc + loss_attr

        return loss, loss_sc, loss_loc, loss_attr
