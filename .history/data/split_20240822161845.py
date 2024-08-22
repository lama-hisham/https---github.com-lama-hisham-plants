import torch

def split_data(dataset, train_size, val_size, test_size):
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset