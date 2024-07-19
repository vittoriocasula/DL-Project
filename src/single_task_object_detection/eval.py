import torch
from config_experiments import config, parse_args
from utils import set_seed, set_device
from dataloader import VOC08Attr
import torchvision.transforms as transforms
from model_compressed import ObjectDetectionModel
from metrics import compute_mAP, view_mAP_for_class
import wandb

if __name__ == "__main__":


    wandb.init(
        group="object_detection",
        project="DL",
        config=config,
        save_code=True,
        mode="disabled",
    )
    model_path = parse_args().model_path
    transform_test = transforms.Compose(
        [
            transforms.Resize(size=config["transform"]["resize_values"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transform"]["mean"], std=config["transform"]["std"]
            ),
        ]
    )
    set_seed(config["global"]["seed"])
    device = set_device(config["global"]["gpu_id"])
    data_test = VOC08Attr(train=False, transform=transform_test)
    model = ObjectDetectionModel().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    mAP = compute_mAP(data_test, model, device)
    view_mAP_for_class(mAP, data_test)

