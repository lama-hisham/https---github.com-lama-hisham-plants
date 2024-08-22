# main/main.py

import torch
import os
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from training.trainer import train_model
from data.preprocessed import get_preprocessed_transform
from data.split import split_data

# Set the dataset path and model parameters
dataset_path = r'C:\Users\Lenovo\Downloads\extractedcitrus\Citrus'
num_classes = 20  # number of plant disease classes

# Define transformations for data preprocessing
transform = get_preprocessed_transform()

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Ensure all data is used

# Create indices for each split
indices = list(range(total_size))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Create subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# Modify the final layer to match the number of classes in your dataset
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define number of epochs
num_epochs = 5

# Train the model
train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)


