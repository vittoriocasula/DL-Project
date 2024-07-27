import torchvision
import torch
import torch.nn as nn
from torchvision.models import AlexNet_Weights
from config_experiments import config
from bbox_transform import regr_to_bbox


def truncated_svd_decomposition(layer, t):
    """
    Applies truncated SVD decomposition on the given linear layer.

    Args:
    - layer (nn.Linear): The linear layer to be decomposed.
    - t (int): Number of singular values to keep.

    Returns:
    - nn.Sequential: A sequential model with two linear layers representing the truncated SVD.
    """

    # Perform SVD on the weight matrix
    W = layer.weight.data
    U, S, V = torch.svd(W)

    # Keep only the top t singular values
    U_t = U[:, :t]
    S_t = S[:t]
    V_t = V[:, :t]

    # Create the two new linear layers
    first_layer = nn.Linear(V_t.shape[0], t, bias=False)
    second_layer = nn.Linear(t, U_t.shape[0], bias=True)

    # Initialize the weights of the new layers
    first_layer.weight.data = (V_t * S_t).t()
    second_layer.weight.data = U_t

    # Set the bias of the second layer to be the same as the original layer
    second_layer.bias.data = layer.bias.data.clone()

    # Return a sequential model of the two layers
    return nn.Sequential(first_layer, second_layer)


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

        original_classifier = nn.Sequential(
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

        # Apply truncated SVD
        self.classifier = nn.Sequential(
            original_classifier[0],
            truncated_svd_decomposition(original_classifier[1], t=256),
            original_classifier[2],
            original_classifier[3],
            truncated_svd_decomposition(original_classifier[4], t=256),
            original_classifier[5],
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
        # Apply truncated SVD
        self.cls_score = truncated_svd_decomposition(self.cls_score, t=256)
        self.bbox = truncated_svd_decomposition(self.bbox, t=256)

        nn.init.normal_(self.cls_score[1].weight, mean=0, std=0.01)
        nn.init.normal_(self.bbox[1].weight, mean=0, std=0.001)
        nn.init.constant_(self.cls_score[1].bias, 0)
        nn.init.constant_(self.bbox[1].bias, 0)

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
        # bboxs = torchvision.ops.clip_boxes_to_image(bboxs, (heigth, width))
        # bboxs = bboxs.view(-1, (config["global"]["num_classes"] + 1), 4)
        # classes = cls_max_score.view(-1, 1, 1).expand(cls_max_score.size(0), 1, 4)
        # bboxs = bboxs.gather(1, classes).squeeze(1)
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
        """
        lbl = labels.view(-1, 1, 1).expand(
            labels.size(0), 1, 4
        )  # view --> 128,1,1 expand --> 128, 1, 4
        # ignore background
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        """
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
