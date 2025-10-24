# src/runner.py
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path

# --- imports ---
from data.loader import get_dataloaders
from models.resnet import build_resnet18
from models.alexnet import build_alexnet
from models.vgg import build_vgg
from train.train_loop import train_one_epoch, evaluate
from utils.helpers import save_checkpoint


def main(config):
    print("✅ Starting training with config:")
    print(config)

    wandb.init(project=config["wandb"]["project"], config=config, name=config["wandb"]["run_name"])

    device = config["device"]
    train_cfg = config["training"]

    # --- dataset ---
    trainloader, testloader = get_dataloaders(batch_size=train_cfg["batch_size"])

    # --- model selection ---
    model_name = config["model"].lower()
    if model_name in ["resnet", "resnet18"]:
        model = build_resnet18(num_classes=10)
    elif model_name == "alexnet":
        model = build_alexnet(num_classes=10, dropout=train_cfg.get("dropout", 0.5))
    elif model_name == "vgg":
        model = build_vgg(num_classes=10, dropout=train_cfg.get("dropout", 0.5))
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    model = model.to(device)

    # --- loss & optimizer ---
    criterion = nn.CrossEntropyLoss()

    if train_cfg["optimizer"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    elif train_cfg["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_cfg["lr"], momentum=0.9, weight_decay=train_cfg["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg['optimizer']}")

    # --- training loop ---
    best_acc = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_acc, train_loss = train_one_epoch(model, trainloader, optimizer, criterion, device)
        val_acc, val_loss = evaluate(model, testloader, criterion, device)

        wandb.log({
            "epoch": epoch,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_loss": val_loss,
        })

        print(f"Epoch [{epoch}/{train_cfg['epochs']}]: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, config["experiment_dir"], best_acc)

    print(f"✅ Training complete. Best Val Acc: {best_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
