import torch
from config_experiments import config
from utils import set_seed, set_device, log_dict
from dataloader import VOC08Attr, collate_fn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from bbox_transform import relative_to_absolute_bbox
import wandb
from model import ObjectDetectionModel
from tqdm import tqdm
from metrics import compute_mAP, view_mAP_for_class
import os
import datetime
import logging


def train(data_loader, model, optimizer, device):
    model.train()

    losses_total = []
    losses_cls = []
    losses_loc = []
    data_loader = tqdm(data_loader, desc="Training", leave=False)

    for i, (image, train_roi, train_cls, train_offset, _, indices_batch) in enumerate(
        data_loader
    ):

        image = image.to(device)
        train_roi = train_roi.to(device)
        train_cls = train_cls.to(device)
        train_offset = train_offset.to(device)
        indices_batch = indices_batch.to(device)
        _, _, height, width = image.shape

        train_roi = relative_to_absolute_bbox(
            boxes=train_roi, image_size=(width, height)
        )
        cls_score, bbox_offset = model(image, train_roi, indices_batch)
        total_loss, loss_cls, loss_loc = model.calc_loss(
            cls_score, bbox_offset, train_cls, train_offset
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses_total.append(total_loss)
        losses_cls.append(loss_cls)
        losses_loc.append(loss_loc)

        wandb.log(
            {
                "loss": total_loss.item(),
                "loss_cls": loss_cls.item(),
                "loss_loc": loss_loc.item(),
            }
        )
        # Updating tqdm description with loss values
        data_loader.set_postfix(
            {
                "Total Loss": total_loss.item(),
                "Cls Loss": loss_cls.item(),
                "Loc Loss": loss_loc.item(),
            }
        )
    return (
        torch.mean(torch.tensor(losses_total)).item(),
        torch.mean(torch.tensor(losses_cls)).item(),
        torch.mean(torch.tensor(losses_loc)).item(),
    )


def eval(data_loader, model, device):
    model.eval()
    losses_total = []
    losses_cls = []
    losses_loc = []
    data_loader = tqdm(data_loader, desc="Evaluation", leave=False)

    for i, (image, train_roi, train_cls, train_offset, _, indices_batch) in enumerate(
        data_loader
    ):

        image = image.to(device)
        train_roi = train_roi.to(device)
        train_cls = train_cls.to(device)
        train_offset = train_offset.to(device)
        indices_batch = indices_batch.to(device)
        _, _, height, width = image.shape

        train_roi = relative_to_absolute_bbox(
            boxes=train_roi, image_size=(width, height)
        )
        cls_score, bbox_offset = model(image, train_roi, indices_batch)
        total_loss, loss_cls, loss_loc = model.calc_loss(
            cls_score, bbox_offset, train_cls, train_offset
        )

        losses_total.append(total_loss)
        losses_cls.append(loss_cls)
        losses_loc.append(loss_loc)

        wandb.log(
            {
                "loss": total_loss.item(),
                "loss_cls": loss_cls.item(),
                "loss_loc": loss_loc.item(),
            }
        )
        # Updating tqdm description with loss values
        data_loader.set_postfix(
            {
                "Total Loss": total_loss.item(),
                "Cls Loss": loss_cls.item(),
                "Loc Loss": loss_loc.item(),
            }
        )
    return (
        torch.mean(torch.tensor(losses_total)).item(),
        torch.mean(torch.tensor(losses_cls)).item(),
        torch.mean(torch.tensor(losses_loc)).item(),
    )


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    logging.basicConfig(
        filename=f"./outputs/{current_time}.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s ",
    )
    log_dict(config)

    set_seed(config["reproducibility"]["seed"])
    device = set_device(config["device"]["gpu_id"])

    model_name = f"model_{current_time}"

    wandb.init(
        group="object_detection",
        project="DL",
        config=config,
        save_code=True,
        # mode="disabled",
    )

    transform_train = transforms.Compose(
        [
            transforms.Resize(
                size=config["transform"]["resize_values"],
                max_size=config["transform"]["max_size"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transform"]["mean"], std=config["transform"]["std"]
            ),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(
                size=config["transform"]["resize_values"],
                max_size=config["transform"]["max_size"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["transform"]["mean"], std=config["transform"]["std"]
            ),
        ]
    )
    # dataset & dataloader

    train_data = VOC08Attr(train=True, transform=transform_train)
    val_data = VOC08Attr(train=False, transform=transform_val)

    train_dataloader = DataLoader(
        train_data,
        batch_size=config["preprocessing"]["n_images"],
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config["preprocessing"]["n_images"], collate_fn=collate_fn
    )

    # model
    model = ObjectDetectionModel().to(device)
    # optimizer
    params = []
    for name, param in model.named_parameters():
        if "weight" in name:
            params.append(
                {
                    "params": param,
                    "lr": config["optimizer"]["lr_global"]
                    * config["optimizer"]["lr_weigth_mult"],
                }
            )
        elif "bias" in name:
            params.append(
                {
                    "params": param,
                    "lr": config["optimizer"]["lr_global"]
                    * config["optimizer"]["lr_bias_bias"],
                }
            )

    optimizer = torch.optim.SGD(
        params,
        lr=config["optimizer"]["lr_global"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # training loop
    best_mAP = 0.0
    best_mAP_dict = {}
    best_epoch = 0
    sched = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["optimizer"]["sched_step_size"],
        gamma=config["optimizer"]["sched_gamma"],
    )

    for epoch in range(config["training_loop"]["num_epochs"]):
        train_loss_total, train_loss_class, train_loss_loc = train(
            train_dataloader, model, optimizer, device
        )
        mAP_train_dict = compute_mAP(train_data, model, device)  # mAP on train
        val_loss_total, val_loss_class, val_loss_loc = eval(
            val_dataloader, model, device
        )
        mAP_val_dict = compute_mAP(val_data, model, device)  # mAP on val

        mAP_train = mAP_train_dict["map_50"].item()
        mAP_val = mAP_val_dict["map_50"].item()

        logging.info(
            f"[Epoch {epoch+1}]\tTRAIN: total_loss = {train_loss_total:.4f} loss_cls = {train_loss_class:.4f} loss_loc = {train_loss_loc:.4f} \tVAL: total_loss val= {val_loss_total:.4f} loss_cls = {val_loss_class:.4f} loss_loc = {val_loss_loc:.4f} \t mAP_train: {mAP_train*100:.2f} mAP_val: {mAP_val*100:.2f}"
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "loss training": train_loss_total,
                "loss validation": val_loss_total,
                "loss classification training": train_loss_class,
                "loss classification validation": val_loss_class,
                "loss location training": train_loss_loc,
                "loss location validation": val_loss_loc,
                "mAP_train": mAP_train,
                "mAP_val": mAP_val,
            }
        )
        sched.step()

        if mAP_val > best_mAP:
            best_model_weights = model.state_dict()
            best_epoch = epoch + 1
            best_mAP = mAP_val
            best_mAP_dict = mAP_val_dict

    logging.info(f"Best model found at epoch {best_epoch}")
    wandb.config.update({"best_epoch": best_epoch})

    # storage model

    path_models = "./models/object_detection/"
    os.makedirs(path_models, exist_ok=True)
    torch.save(best_model_weights, path_models + model_name + ".pth")

    view_mAP_for_class(best_mAP_dict, val_data)

    wandb.finish()
