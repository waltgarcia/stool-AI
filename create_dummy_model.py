#!/usr/bin/env python3
"""
Script to create a dummy model weights file for testing purposes
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

class BristolClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(BristolClassifier, self).__init__()
        try:
            self.backbone = models.resnet50(weights=None)
        except Exception:
            self.backbone = models.resnet50(pretrained=False)

        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create and save model
print("Creating dummy model...")
model = BristolClassifier(num_classes=7)
state_dict = model.state_dict()

# Save weights
torch.save(state_dict, "model_weights.pth")
print("✅ Model weights saved to 'model_weights.pth'")
