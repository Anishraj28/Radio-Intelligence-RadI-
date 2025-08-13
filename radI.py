import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from torch.utils.data import random_split


# To use mac gpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")






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



def main():
        
    # Convert image files into 3-D tensor

    # Our training data should differentiate as much as possible for higher accuracy
    # This is why we adjust the tensors differently
    # Agumentation
    train_transform = transforms.Compose([
        transforms.Resize((275, 275)),  # 275x275 pixels
        transforms.RandomCrop(250),     # Then random crop to 250x250
        transforms.RandomHorizontalFlip(0.5),  # 50% chance to flip left-right
        transforms.RandomRotation(7),   # Rotate 5 degrees
        transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),  # Vary brightness/contrast
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Test transform (no augmentation, basic preprocessing)
    test_transform = transforms.Compose([
        transforms.Resize((275, 275)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = datasets.ImageFolder(root="/Users/anishrajumapathy/Downloads/chest_xray/train", transform=train_transform)
    test_data = datasets.ImageFolder(root="/Users/anishrajumapathy/Downloads/chest_xray/test", transform=test_transform)

    '''
    # Train and test data... since we dont have seperate folders for train and test, we have to do a random split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    '''

    # Create a snall batch size for images.....
    train_loader = DataLoader(train_data, batch_size = 16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size = 16, shuffle=True, num_workers=4, pin_memory=True)

    # Print info about your datasets
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {train_data.classes}")


    '''
    # Define Our CNN Model
    #Describe convolutional layer and whats its doing (2 Layers)

    # you can set padding so that the image doesn't get smaller in pixel size (needed for real world stuff)
    conv1 = nn.Conv2d(3, 6, 3, 1, 1)
    conv2 = nn.Conv2d(6, 16, 3, 1, 1)

    # Grab 1 x-ray
    for i, (X_Train, y_train) in enumerate(train_data):
        break

    X_Train.shape

    # This will change tensor form three chanels to 4 since we need to have batch size as well
    x = X_Train.unsqueeze(0)

    # Perform our first convolution
    x = F.relu(conv1(x)) # Rectified Linear Unit for our activation function

    # [1 single image, 6 is filters, 2600, 2600 is dimension of image]
    x.shape

    # pass through pooling layer
    x = F.max_pool2d(x,2,2) # kernal of 2 and stride of 2

    # Do our second convolutional layer
    x = F.relu(conv2(x))
    x.shape # to view....same as before

    # Do another pooling layer 
    x = F.max_pool2d(x,2,2)
    x.shape 
    '''

    import time
    start_time = time.time()



        
    torch.manual_seed(41)
    model = ConvolutionalNetwork().to(device)
    model

    # Loss Function Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay=1e-4) # Small Learning rate , slower but accurate....bigger learning rate is vice versa / weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2) # Learning rate scheduler


    # Create variables to track things
    epochs = 15
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []


    for i in range(epochs):
        trn_corr = 0
        tst_corr=0
        #Train

        for b,(X_train, y_train) in enumerate(train_loader):
            b+=1 # start our batches at 1
            
            # Move batch to GPU
            X_train, y_train = X_train.to(device), y_train.to(device)
            
            y_pred = model(X_train) # get predicted values from the training set, Not flattened
            loss = criterion(y_pred, y_train) # how off are we? Compare the perdiction to correct
            
            predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions. Indexed off the first point
            batch_corr = (predicted == y_train).sum() # how many we got correct from this batch. True = 1, False = 0
            trn_corr += batch_corr # keep track as we go along in training
        
            # Update Parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print Results
            if b%10 == 0:
                print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

        train_losses.append(loss.item())
        train_correct.append(trn_corr.item())
        
        # Test
        with torch.no_grad(): # No gradient so we dont update our weights and biasis with test data
            test_loss_total = 0  # initialize before the loop
            for b,(X_test, y_test) in enumerate(test_loader):
                # Move batch to GPU
                X_test, y_test = X_test.to(device), y_test.to(device)
                y_val = model(X_test)
                
                loss = criterion(y_val, y_test)
                test_loss_total += loss.item()  # accumulate loss
                
                predicted = torch.max(y_val.data, 1)[1] # Adding up correct predictions
                tst_corr += (predicted == y_test).sum() # True = 1, F = 0 and sum away
            
            avg_test_loss = test_loss_total / len(test_loader)
            test_losses.append(avg_test_loss)
            test_correct.append(tst_corr.item())
        
    # to save model
    torch.save(model.state_dict(), 'trained_xray_model.pth')
    print("Model saved as 'trained_xray_model.pth'!")

    print(f"Training accuracy: {train_correct[-1]/len(train_data)*100:.2f}%")
    print(f"Test accuracy: {test_correct[-1]/len(test_data)*100:.2f}%")
    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")

    current_time = time.time()
    total = current_time - start_time
    print(f'Training Took: {total/60} minutes!!!')
    
if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
    