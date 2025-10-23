# src/models/resnet.py
import torch.nn as nn
import torchvision.models as models

def build_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
