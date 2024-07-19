import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config_experiments import config
from utils import set_seed, set_device, log_dict
from dataloader import VOC08Attr
from model import AttributePredictionModel
from metrics import compute_mAP, mAP_view_attributes

import wandb
from tqdm import tqdm
import os
import datetime
import logging


def train(data_loader, model, optimizer, device):
    model.train()

    losses_total = []

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

        attr_score = model(image, train_roi, indices_batch)
        total_loss = model.calc_loss(attr_score, train_cls, train_attr)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses_total.append(total_loss)

        wandb.log(
            {
                "loss": total_loss.item(),
            }
        )
        # Updating tqdm description with loss values
        data_loader.set_postfix(
            {
                "Total Loss": total_loss.item(),
            }
        )

    return torch.mean(torch.tensor(losses_total)).item()


def eval(data_loader, model, device):
    model.eval()
    losses_total = []

    data_loader = tqdm(data_loader, desc="Evaluation", leave=False)

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

        attr_score = model(image, train_roi, indices_batch)
        total_loss = model.calc_loss(attr_score, train_cls, train_attr)

        losses_total.append(total_loss)

        wandb.log(
            {
                "loss": total_loss.item(),
            }
        )
        # Updating tqdm description with loss values
        data_loader.set_postfix(
            {
                "Total Loss": total_loss.item(),
            }
        )
    return torch.mean(torch.tensor(losses_total)).item()



if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    logging.basicConfig(
        filename=f"./outputs/attribute_prediction/{current_time}.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s ",
    )

    log_dict(config)

    set_seed(config["global"]["seed"])
    device = set_device(config["global"]["gpu_id"])

    model_name = f"model_{current_time}"

    wandb.init(
        group="attribute_prediction",
        project="DL",
        config=config,
        save_code=True,
        # mode="disabled",
    )
    wandb.config.update({"model_name": model_name})

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
    )

    # model
    model = AttributePredictionModel().to(device)
    if config["model"].get("load_pretrained_model"):
        logging.info("load %s\n" % config["model"]["load_pretrained_model"])
        model.load_state_dict(torch.load(config["model"]["load_pretrained_model"]))
    else:
        logging.info(
            "Pretrained models not provided. Uncomment load_pretrained_model in YAML file and insert model's path or the model will be randomly initialized."
        )

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

    for epoch in range(config["global"]["num_epochs"]):
        train_loss_total = train(
            train_dataloader, model, optimizer, device
        )
        # mAP_train_dict = compute_mAP(train_data, model, device)  # mAP on train
        """
        val_loss_total, val_loss_class, val_loss_loc = eval(
            val_dataloader, model, device
        )"""
        if epoch % 5 == 0:
            mAP_val_dict = compute_mAP(val_data, model, device)  # mAP on val
            #mAP_val_dict = torch.nan_to_num(mAP_val_dict, nan=0.0)
            mAP_val = torch.mean(mAP_val_dict).item()


            """l
            ogging.info(
                f"[Epoch {epoch+1}]\tTRAIN: total_loss = {train_loss_total:.4f} loss_cls = {train_loss_class:.4f} loss_loc = {train_loss_loc:.4f} \tVAL: total_loss val= {val_loss_total:.4f} loss_cls = {val_loss_class:.4f} loss_loc = {val_loss_loc:.4f} \t mAP_train: {mAP_train*100:.2f} mAP_val: {mAP_val*100:.2f}"
            )"""

            logging.info(
                f"[Epoch {epoch+1}]\tTRAIN: total_loss = {train_loss_total:.4f} \t mAP_val: {mAP_val*100:.2f}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss training": train_loss_total,
                    # "loss validation": val_loss_total,
                    # "loss classification validation": val_loss_class,
                    # "loss location validation": val_loss_loc,
                    # "mAP_train": mAP_train,
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

    path_models = "./models/attribute_prediction/"
    os.makedirs(path_models, exist_ok=True)
    torch.save(best_model_weights, path_models + model_name + ".pth")

    mAP_view_attributes(best_mAP_dict, val_data)

    wandb.finish()
