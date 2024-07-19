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
            output_size=(
                config["model"]["output_size_roipool"][0],
                config["model"]["output_size_roipool"][1],
            ),
            spatial_scale=config["model"]["spatial_scale"],
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(
                256
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

