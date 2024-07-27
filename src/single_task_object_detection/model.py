import torchvision
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights
from config_experiments import config
from bbox_transform import regr_to_bbox
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        rawnet = torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)
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
        cel = nn.CrossEntropyLoss(weight=self.get_class_weigths().to(labels.device), label_smoothing=0.1)
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

    def get_class_weigths(self,):
        instances_class = np.concatenate(
            [
                np.repeat(7, 184),
                np.repeat(8, 161),
                np.repeat(16, 254),
                np.repeat(9, 232),
                np.repeat(3, 297),
                np.repeat(19, 73),
                np.repeat(6, 533),
                np.repeat(14, 187),
                np.repeat(10, 447),
                np.repeat(20, 103),
                np.repeat(11, 113),
                np.repeat(4, 244),
                np.repeat(1, 153),
                np.repeat(18, 150),
                np.repeat(2, 2500),
                np.repeat(12, 225),
                np.repeat(17, 123),
                np.repeat(15, 121),
                np.repeat(13, 90),
                np.repeat(5, 150),
            ]
        )
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(instances_class), y=instances_class
        )
        class_weights_dict = dict(zip(np.unique(instances_class), class_weights))
        foreground_weight_mean = np.mean(list(class_weights_dict.values()))
        background_weight = foreground_weight_mean / 10
        class_weights_dict[0] = background_weight
        class_weights_list = [class_weights_dict[i] for i in range(len(class_weights_dict))]
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)
        return class_weights_tensor
