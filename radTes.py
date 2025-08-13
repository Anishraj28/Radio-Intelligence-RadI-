import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from PIL import Image


# To use mac gpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


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
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X,2,2) # 2x2 kernal and a stride of 2
        X = F.dropout(X, 0.2, training=self.training)
        # Second Pass
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X,2,2) # 2x2 kernal and a stride of 2
        # Third Pass
        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X,2,2) # 2x2 kernal and a stride of 2
        # Final Pass
        X = F.relu(self.bn4(self.conv4(X)))
        
        # Add this after conv4
        X = F.max_pool2d(X, X.size()[2:])  # Global Max Pooling
        X = X.view(-1, 256)  # Then use 256 because thats how much channels we have total


        # Fully Connected Layers
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        return X




# Convert image files into 3-D tensor
# Test transform (no augmentation, basic preprocessing)
test_transform = transforms.Compose([
    transforms.Resize((275, 275)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



# Load model
model = ConvolutionalNetwork().to(device)
model.load_state_dict(torch.load('trained_xray_model.pth', weights_only=True))
model.eval()

def predict(img_path):
    # Load and PreProcess Image
    image = Image.open(img_path)
    image.show()
    image = test_transform(image).unsqueeze(0).to(device) # We have to do unsqueeze because we have to have a batch dimension
    
    1
    # Our model is trained using batches on images.....even though our input will only be 1 xray....we have to change the dimensions so that the model thinks its a batch

    # Evaluate with image

    with torch.no_grad():
        # The raw numbers from the model
        output = model(image)
        
        # We have to convert the raw output numbers into percentages
        probabilities = torch.softmax(output, dim=1)  
        
        # Now we have to compare which percentage is highest
        predicted_class = torch.argmax(output, dim=1)
        
        # Index our results 
        classes = ['NORMAL', 'PNEUMONIA']
        prediction = classes[predicted_class.item()]  # Convert index to name
        
        
        # Necessary for medical imaging: Confidence in Prediction
        confidence = probabilities[0][predicted_class].item() * 100
    
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}%")
        return prediction, confidence

predict("/Users/anishrajumapathy/Downloads/chest_xray/val/PNEUMONIA/f054912a.jpg")

