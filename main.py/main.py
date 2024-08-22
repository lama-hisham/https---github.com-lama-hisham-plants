import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .data.loader import get_data_loader, get_preprocessed_transform
from .data.split import split_data
from .model.architecture import define_model
from .training.trainer import train_model
from .evaluation.evaluator import test_model

# Set the dataset path and model parameters
dataset_path = r'C:\Users\Lenovo\Downloads\extractedcitrus\Citrus'
num_classes = 20  # number of plant disease classes

# Preprocess the dataset
transform = get_preprocessed_transform()
dataset = ImageFolder(dataset_path, transform=transform)

# Split the dataset into training, validation, and testing sets
train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(0.5 * val_test_size)         
test_size = val_test_size - val_size
train_dataset, val_dataset, test_dataset = split_data(dataset, train_size, val_size, test_size)

# Create data loaders
train_loader, val_loader, test_loader = get_data_loader(train_dataset, val_dataset, test_dataset, batch_size=32)

# Define the model
model = define_model(num_classes)

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

# Test the model
test_model(model, test_loader)