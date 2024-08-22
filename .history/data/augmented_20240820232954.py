import torchvision.transforms as transforms

def get_augmented_transform():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])
    return transform