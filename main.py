import torch
from torchvision import models
from torch.utils.data import DataLoader
from training.trainer import train_model
from data.augmented import get_train_transform, get_val_transform
from data.split import split_data
from evaluation.evaluator import test_model
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder

# Set the dataset path and model parameters
dataset_path = r'C:\Users\Lama\Downloads\extractedcitrus\Citrus'
num_classes = 20  # number of plant disease classes

# Define transformations for data preprocessing
train_transform = get_train_transform()
val_transform = get_val_transform()

# Load dataset
full_dataset = ImageFolder(root=dataset_path, transform=train_transform)

# Define sizes for splits
train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

# Split the dataset
train_dataset, val_dataset, test_dataset = split_data(full_dataset, train_percentage, val_percentage, test_percentage)

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

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
print("Starting training...")
train_model(model, train_loader, val_loader, num_epochs=3, device=device, optimizer=optimizer, criterion=criterion)

# Evaluate the model on the test dataset
print("Training complete. Evaluating model...")
test_model(model, test_loader, device, criterion)

print("Evaluation complete.")




