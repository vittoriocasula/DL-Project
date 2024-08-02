import torchvision
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights, VGG16_Weights
from config_experiments import config
from bbox_transform import regr_to_bbox
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        rawnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
        # rawnet = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)

        self.features = nn.Sequential(*list(rawnet.features.children())[:-1])

        # Freezare il primo layer conv
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

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                256  # 512
                * config["model"]["output_size_roipool"][0]
                * config["model"]["output_size_roipool"][1],
                4096,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, rois, ridx):
        idx_rois = torch.cat(
            (ridx, rois), dim=-1
        )  # create matrix with (batch_idx, rois(xyxy))
        res = self.roipool(features, idx_rois)
        res = res.view(res.size(0), -1)
        feat = self.classifier(res)
        return feat


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
        self.alex = Backbone()
        self.roi_module = ROI_Module()
        self.attribute_head = AttributePredictionHead(
            num_attributes=config["global"]["num_attributes"]
        )

    def forward(self, x, rois, ridx):
        out = self.alex(x)
        out = self.roi_module(out, rois, ridx)
        score_attr = self.attribute_head(out)
        return score_attr

    def prediction_rois(self, img, rois, ridx):
        self.eval()
        with torch.no_grad():
            sc_attr = self(img, rois, ridx)
        score_attr = nn.functional.sigmoid(sc_attr)
        attr = torch.where(score_attr > 0.5, torch.tensor(1.0, device = sc_attr.device), torch.tensor(0.0, device = sc_attr.device))
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
