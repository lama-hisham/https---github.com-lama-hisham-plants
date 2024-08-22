import torch
import os
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from training.trainer import train_model
 # Import from the training folder

# Set the dataset path and model parameters
dataset_path = r'C:\Users\Lenovo\Downloads\extractedcitrus\Citrus'
num_classes = 20  # number of plant disease classes


# Define transformations for data preprocessing
transform = get_preprocessed_transform()

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training, validation, and testing sets
train_size = int(0.8 * len(dataset))
val_test_size = len(dataset) - train_size
val_size = int(0.5 * val_test_size)         
test_size = val_test_size - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained AlexNet model
model = models.alexnet(pretrained=True)

# Modify the final layer to match the number of classes in your dataset
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define number of epochs
num_epochs = 5

# Train the model
train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
