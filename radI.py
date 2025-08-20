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
from PIL import Image

# Device setup for Apple Silicon (M3 chip)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class CenterCropPercent:
    def __init__(self, percent=0.2):
        self.percent = percent

    def __call__(self, img):
        w, h = img.size
        dx, dy = int(w * self.percent), int(h * self.percent)
        # Crop box: (left, top, right, bottom)
        return img.crop((dx, dy, w - dx, h - dy))

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


def main():
    # Image transformations
    IMG_SIZE = 900
    CROP_SIZE = 1800  # smaller than 2000, cuts out text edges

    train_transform = transforms.Compose([
        CenterCropPercent(0.1),  # remove 10% border
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        CenterCropPercent(0.1),  # remove 10% border
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load datasets
    train_data = datasets.ImageFolder(
        root="/Users/anishrajumapathy/Downloads/chest_xray/train",
        transform=train_transform
    )
    test_data = datasets.ImageFolder(
        root="/Users/anishrajumapathy/Downloads/chest_xray/test",
        transform=test_transform
    )

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {train_data.classes}")

    # Model
    torch.manual_seed(42)
    model = ConvolutionalNetwork().to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Tracking
    epochs = 30
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        trn_corr, train_loss_total = 0, 0
        for b, (X_train, y_train) in enumerate(train_loader, start=1):
            X_train, y_train = X_train.to(device), y_train.to(device)

            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            predicted = torch.max(y_pred, 1)[1]
            trn_corr += (predicted == y_train).sum().item()

            if b % 10 == 0:
                print(f'Epoch {epoch+1} Batch {b} Loss: {loss.item():.4f}')

        avg_train_loss = train_loss_total / len(train_loader)
        train_accuracy = trn_corr / len(train_data)
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # Testing
        model.eval()
        tst_corr, test_loss_total = 0, 0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_val = model(X_test)

                loss = criterion(y_val, y_test)
                test_loss_total += loss.item()

                predicted = torch.max(y_val, 1)[1]
                tst_corr += (predicted == y_test).sum().item()

        avg_test_loss = test_loss_total / len(test_loader)
        test_accuracy = tst_corr / len(test_data)
        test_losses.append(avg_test_loss)
        test_accs.append(test_accuracy)

        scheduler.step(avg_test_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.4f}")

    # Save final model
    torch.save(model.state_dict(), 'trained_xray_model.pth_3')
    print("Model saved as 'trained_xray_model.pth_3'!")

    total_time = (time.time() - start_time) / 60
    print(f"Training Took: {total_time:.2f} minutes!")

    # Grad-CAM
    print("\n" + "="*50)
    print("Starting Grad-CAM Analysis...")
    gradcam.analyze_batch_gradcam(model, test_loader, device, num_samples=5)


if __name__ == "__main__":
    main()
