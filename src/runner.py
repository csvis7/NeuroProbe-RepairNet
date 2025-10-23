# src/runner.py
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pathlib import Path

from data.loader import get_dataloaders
from models.resnet import build_resnet18
from train.train_loop import train_one_epoch, evaluate
from utils.helpers import save_checkpoint

def main(config):
    print("✅ Starting training with config:")
    print(config)

    wandb.init(project=config["wandb"]["project"], config=config, name=config["wandb"]["run_name"])

    device = config["device"]
    train_cfg = config["training"]

    trainloader, testloader = get_dataloaders(batch_size=train_cfg["batch_size"])
    model = build_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

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
