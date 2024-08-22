# data/split.py

import torch

def split_data(dataset, train_size, val_size, test_size):
    # Create a random split based on the provided sizes
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
