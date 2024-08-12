import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config_experiments import config
from dataloader import VOC08Attr
from model import CrossStitchNet, ObjectDetectionModel, AttributePredictionModel
from utils import set_seed, set_device, log_dict, get_map_step, copy_weights

from metrics import (
    compute_mAP_obj_detect,
    compute_mAP_attr_predict,
    view_mAP_for_class,
    mAP_view_attributes,
)

import wandb
from tqdm import tqdm
import os
import datetime
import logging


def train(data_loader, model, optimizer, device):
    model.train()

    losses_total = []
    losses_cls = []
    losses_loc = []
    losses_attr = []

    data_loader = tqdm(data_loader, desc="Training", leave=False)

    for i, (
        image,
        train_roi,
        train_cls,
        train_offset,
        train_attr,
        indices_batch,
    ) in enumerate(data_loader):

        image = image.to(device)
        train_roi = train_roi.to(device)
        train_cls = train_cls.to(device)
        train_offset = train_offset.to(device)
        train_attr = train_attr.to(device)
        indices_batch = indices_batch.to(device)

        cls_score, bbox_offset, attr = model(image, train_roi, indices_batch)
        total_loss, loss_cls, loss_loc, loss_attr = model.calc_loss(
            cls_score, bbox_offset, train_cls, train_offset, attr, train_attr
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses_total.append(total_loss)
        losses_cls.append(loss_cls)
        losses_loc.append(loss_loc)
        losses_attr.append(loss_attr)

        wandb.log(
            {
                "loss": total_loss.item(),
                "loss_cls": loss_cls.item(),
                "loss_loc": loss_loc.item(),
                "loss_attr": loss_attr.item(),
            }
        )
        # Updating tqdm description with loss values
        data_loader.set_postfix(
            {
                "Total Loss": total_loss.item(),
                "Cls Loss": loss_cls.item(),
                "Loc Loss": loss_loc.item(),
                "Attr Loss": loss_attr.item(),
            }
        )

    return (
        torch.mean(torch.tensor(losses_total)).item(),
        torch.mean(torch.tensor(losses_cls)).item(),
        torch.mean(torch.tensor(losses_loc)).item(),
        torch.mean(torch.tensor(losses_attr)).item(),
    )


def eval(data_loader, model, device):
    model.eval()
    losses_total = []
    losses_cls = []
    losses_loc = []
    losses_attr = []

    data_loader = tqdm(data_loader, desc="Evaluation", leave=False)
    with torch.no_grad():
        for i, (image, roi, classes, offset, gt_attr, indices_batch) in enumerate(
            data_loader
        ):

            image = image.to(device)
            roi = roi.to(device)
            classes = classes.to(device)
            offset = offset.to(device)
            gt_attr = gt_attr.to(device)
            indices_batch = indices_batch.to(device)

            cls_score, bbox_offset, pred_attr = model(image, roi, indices_batch)
            total_loss, loss_cls, loss_loc, loss_attr = model.calc_loss(
                cls_score, bbox_offset, classes, offset, pred_attr, gt_attr
            )

            losses_total.append(total_loss)
            losses_cls.append(loss_cls)
            losses_loc.append(loss_loc)
            losses_attr.append(loss_attr)

            wandb.log(
                {
                    "loss": total_loss.item(),
                    "loss_cls": loss_cls.item(),
                    "loss_loc": loss_loc.item(),
                    "loss_attr": loss_attr.item(),
                }
            )
            # Updating tqdm description with loss values
            data_loader.set_postfix(
                {
                    "Total Loss": total_loss.item(),
                    "Cls Loss": loss_cls.item(),
                    "Loc Loss": loss_loc.item(),
                    "Attr Loss": loss_attr.item(),
                }
            )

    return (
        torch.mean(torch.tensor(losses_total)).item(),
        torch.mean(torch.tensor(losses_cls)).item(),
        torch.mean(torch.tensor(losses_loc)).item(),
        torch.mean(torch.tensor(losses_attr)).item(),
    )


if __name__ == "__main__":

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"experiments/cross_stitch/{current_time}", exist_ok=True)

    path_models = f"experiments/cross_stitch/{current_time}/models/"
    os.makedirs(path_models, exist_ok=True)

    logging.basicConfig(
        filename=f"./experiments/cross_stitch/{current_time}/config_and_output.log",
        level=logging.INFO,
        format="%(message)s",
    )
    logging.info("\nCONFIG:\n")
    log_dict(config)
    logging.info("\nOUTPUT:\n")

    set_seed(config["global"]["seed"])
    device = set_device(config["global"]["gpu_id"])

    wandb.init(
        group="cross_stitch",
        project="DL",
        config=config,
        save_code=True,
        # mode="disabled",
    )
    wandb.config.update({"experiment_current_time": current_time})

    # dataset & dataloader

    train_data = VOC08Attr(
        train=True,
        transform=transforms.Compose(
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
        ),
    )
    val_data = VOC08Attr(  # test set
        train=False,
        transform=transforms.Compose(
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
        ),
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config["preprocessing"]["n_images"],
        collate_fn=train_data.collate_fn,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=config["preprocessing"]["n_images"],
        collate_fn=val_data.collate_fn,
        shuffle=False,
    )

    if config["model"].get("load_pretrained_model"):
        model = CrossStitchNet().to(device)
        logging.info("load %s\n" % config["model"]["load_pretrained_model"])
        model.load_state_dict(torch.load(config["model"]["load_pretrained_model"]))
    else:
        if config["model"]["single_task_init"]:
            path_best_model_obj = config["model"]["model_obj"]
            path_best_model_attr = config["model"]["model_attr"]
            model_obj = ObjectDetectionModel().to(device)
            model_attr = AttributePredictionModel().to(device)
            model_obj.load_state_dict(
                torch.load(path_best_model_obj, map_location=device)
            )
            model_attr.load_state_dict(
                torch.load(path_best_model_attr, map_location=device)
            )
            model = CrossStitchNet(model_obj.alex, model_attr.alex).to(device)

            """for model_a, model_b in zip(
                model.cross_stitch_net.models_a, model.cross_stitch_net.models_b
            ):
                copy_weights(model_obj.alex.features, model_a)
                copy_weights(model_attr.alex.features, model_b)"""

            # model.cross_stitch_net.models_a.load_state_dict(model_obj.alex.features.state_dict())
            # model.cross_stitch_net.models_b.load_state_dict(model_attr.alex.features.state_dict())

            model.roi_a.load_state_dict(model_obj.roi_module.state_dict())
            model.roi_b.load_state_dict(model_attr.roi_module.state_dict())

            model.model_obj_detect.load_state_dict(
                model_obj.obj_detect_head.state_dict()
            )
            model.model_attribute.load_state_dict(
                model_attr.attribute_head.state_dict()
            )
        else:
            model = CrossStitchNet(
                ObjectDetectionModel().to(device).alex,
                AttributePredictionModel().to(device).alex,
            ).to(device)
            logging.info(
                "Pretrained models not provided. Uncomment load_pretrained_model in YAML file and insert model's path or the model will be randomly initialized."
            )

    # optimizer
    params = []
    cross_stitch_params = []

    for name, param in model.named_parameters():
        if "alfa_a" in name or "alfa_b" in name:
            cross_stitch_params.append(
                {"params": param, "lr": config["cross_stitch"]["lr_cross_stitch"]}
            )
        if "weight" in name:
            params.append(
                {
                    "params": param,
                    "lr": config["optimizer"]["lr_global"]
                    * config["optimizer"]["lr_weigth_mult"],
                    "weight_decay": config["optimizer"]["weight_decay"],
                }
            )
        elif "bias" in name:
            params.append(
                {
                    "params": param,
                    "lr": config["optimizer"]["lr_global"]
                    * config["optimizer"]["lr_bias_mult"],
                    "weight_decay": config["optimizer"]["weight_decay"],
                }
            )

    optimizer = torch.optim.SGD(
        params,
        momentum=config["optimizer"]["momentum"],
    )
    for param_group in cross_stitch_params:
        optimizer.add_param_group(
            {
                "params": param_group["params"],
                "lr": config["cross_stitch"]["lr_cross_stitch"],
            }
        )

    # training loop
    best_mAP = 0.0
    best_mAP_dict_obj = {}
    best_mAP_dict_attr = {}
    best_epoch = 0
    sched = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["optimizer"]["sched_step_size"],
        gamma=config["optimizer"]["sched_gamma"],
    )

    for epoch in range(config["global"]["num_epochs"]):
        train_loss_total, train_loss_class, train_loss_loc, train_loss_attr = train(
            train_dataloader, model, optimizer, device
        )
        val_loss_total, val_loss_class, val_loss_loc, val_loss_attr = eval(
            val_dataloader, model, device
        )
        logging.info(
            f"[Epoch {epoch+1}]\tTRAIN: total_loss = {train_loss_total:.4f} loss_cls = {train_loss_class:.4f} loss_loc = {train_loss_loc:.4f} loss_attr = {train_loss_attr:.4f} \tVAL: total_loss val= {val_loss_total:.4f} loss_cls = {val_loss_class:.4f} loss_loc = {val_loss_loc:.4f} loss_attr = {val_loss_attr:.4f}"
        )
        map_step = get_map_step(
            epoch, config["global"]["num_epochs"], config["global"]["metrics_step"]
        )
        if (epoch + 1) % map_step == 0:
            mAP_train_dict_obj = compute_mAP_obj_detect(
                train_data, model, device
            )  # mAP on train
            mAP_train_dict_attr = compute_mAP_attr_predict(
                train_data, model, device
            )  # mAP on train
            mAP_val_dict_obj = compute_mAP_obj_detect(
                val_data, model, device
            )  # mAP on validation
            mAP_val_dict_attr = compute_mAP_attr_predict(
                val_data, model, device
            )  # mAP on validation

            mAP_train_obj = mAP_train_dict_obj["map_50"].item()
            mAP_val_obj = mAP_val_dict_obj["map_50"].item()
            mAP_train_attr = torch.mean(mAP_train_dict_attr).item()
            mAP_val_attr = torch.mean(mAP_val_dict_attr).item()

            logging.info(
                f"[Epoch {epoch+1}]\tTRAIN: total_loss = {train_loss_total:.4f} loss_cls = {train_loss_class:.4f} loss_loc = {train_loss_loc:.4f} \tVAL: total_loss val= {val_loss_total:.4f} loss_cls = {val_loss_class:.4f} loss_loc = {val_loss_loc:.4f} \t mAP_train_obj: {mAP_train_obj*100:.2f} mAP_val_obj: {mAP_val_obj*100:.2f} mAP_train_attr: {mAP_train_attr*100:.2f} mAP_val_attr: {mAP_val_attr*100:.2f}"
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
                    "loss attribute training": train_loss_attr,
                    "loss attribute validation": val_loss_attr,
                    "mAP_train_obj": mAP_train_obj,
                    "mAP_val_obj": mAP_val_obj,
                    "mAP_train_attr": mAP_train_attr,
                    "mAP_val_attr": mAP_val_attr,
                }
            )

            # Save model at each epoch
            model_path = os.path.join(path_models, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)

            # prossisoriamente cerchiamo di aumentare le performance dell'object detector
            mAP_val = mAP_val_obj

            if mAP_val > best_mAP:
                best_model_weights = model.state_dict()
                best_epoch = epoch + 1
                best_mAP = mAP_val
                best_mAP_dict_obj = mAP_val_dict_obj
                best_mAP_dict_attr = mAP_val_dict_attr

        else:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss training": train_loss_total,
                    "loss validation": val_loss_total,
                    "loss classification training": train_loss_class,
                    "loss classification validation": val_loss_class,
                    "loss location training": train_loss_loc,
                    "loss location validation": val_loss_loc,
                    "loss attribute training": train_loss_attr,
                    "loss attribute validation": val_loss_attr,
                }
            )
        sched.step()

    logging.info(f"\nBest model found at epoch {best_epoch}")
    wandb.config.update({"best_epoch": best_epoch})

    # storage best model
    torch.save(best_model_weights, f"{path_models}best_model_epoch_{best_epoch}.pth")

    view_mAP_for_class(best_mAP_dict_obj, val_data)
    mAP_view_attributes(mAP_val_dict_attr, val_data)

    wandb.finish()
