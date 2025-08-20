import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import gradcam


# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Adaptive Pooling so we can have heatmap work properly
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Output size: 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X, 2, 2)
        X = F.dropout(X, 0.2, training=self.training)

        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, 2, 2)

        X = F.relu(self.bn4(self.conv4(X)))
        
        # This preserves some spatial structure (4x4 instead of 1x1)
        X = self.adaptive_pool(X)
        
        X = X.view(X.size(0), -1)  # Flatten: [batch, 256*4*4]

        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return X



device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = ConvolutionalNetwork().to(device)
model.load_state_dict(torch.load('trained_xray_model.pth_2', map_location=device, weights_only=True))

# Test on a single image
img_path = '/Users/anishrajumapathy/Downloads/chest_xray/val/PNEUMONIA/pneumonia-145018_915x430.webp'
gradcam.test_single_image(img_path, model, device)