# src/train/train_loop.py
import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0

    for inputs, targets in tqdm(trainloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    loss = running_loss / len(trainloader)
    return acc, loss


def evaluate(model, testloader, criterion, device):
    model.eval()
    total, correct, test_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    loss = test_loss / len(testloader)
    return acc, loss
