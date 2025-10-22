# src/runner.py
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from pathlib import Path

# Utility: Set seeds for reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Training and Evaluation
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ---------------------------
# Main Training Routine
# ---------------------------
def main(config):
    print("✅ Starting training with config:")
    print(config)

    # Init W&B
    wandb.init(project=config["wandb"]["project"], config=config, name=config["wandb"]["run_name"])

    # Setup device and seeds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.get("seed", 42))

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Datasets and loaders
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=2)

    # Model, loss, optimizer, scheduler
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Experiment folder
    exp_dir = Path(config["experiment_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{config['training']['epochs']}]: "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), exp_dir / "best_model.pt")

    print(f"✅ Training complete. Best Val Acc: {best_acc:.2f}%")
    wandb.finish()


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)