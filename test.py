import torch
from torchvision import datasets, transforms
from evaluation.evaluator import test_model
from model.architecture import define_model
from data.loader import get_data_loader
from data.preprocessed import get_preprocessed_transform

# Define the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset_path = r'C:\Users\Lama\Downloads\extractedcitrus\Citrus'
test_transform = get_preprocessed_transform()
test_dataset = datasets.ImageFolder(test_dataset_path, transform=test_transform)

# Create a data loader for the test dataset
test_loader = get_data_loader(test_dataset_path, batch_size=32, transform=test_transform)

# Define the model
num_classes = len(test_dataset.classes)
model = define_model(num_classes)

# Load the trained model weights 
model.load_state_dict(torch.load('model.pth', map_location=device))

# Move the model to the device (GPU or CPU)
model.to(device)
# Test the model on the test dataset
test_model(model, test_loader)

# Calculate the accuracy for each class/disease
class_accuracies = {}
for i, class_name in enumerate(test_dataset.classes):
    class_accuracies[class_name] = 0
    class_correct = 0
    class_total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            class_correct += (predicted == labels).sum().item()
            class_total += labels.size(0)
    class_accuracies[class_name] = class_correct / class_total * 100

# Print the total accuracy and accuracy for each class/disease
print(f'Total Accuracy: {test_model(model, test_loader)}')
print('Class Accuracies:')
for class_name, accuracy in class_accuracies.items():
    print(f'{class_name}: {accuracy:.2f}%')