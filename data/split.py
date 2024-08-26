import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def split_data(dataset, train_percentage, val_percentage, test_percentage):
    total_size = len(dataset)
    print(f"Total dataset size: {total_size}")

    # Calculate sizes based on percentages
    train_size = int(train_percentage * total_size)
    val_size = int(val_percentage * total_size)
    test_size = total_size - train_size - val_size  # Ensures all data is used

    print(f"Calculated sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Split the dataset
    try:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        print("Data split successfully.")
    except Exception as e:
        print("Error during split:", e)
        return None, None, None

    # Print sizes of splits
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

