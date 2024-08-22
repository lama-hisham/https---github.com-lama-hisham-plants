import os
import imageio.v2 as imageio
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

image_folder = r'C:\Users\Lenovo\Downloads\extractedcitrus\Citrus'

images_pil = []
for root, dirs, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.PNG', '.png', '.JPG', '.jpg', '.JPEG', '.jpeg', '.gif', '.GIF', '.bmp', '.BMP', '.TIF', '.tif', '.jfif', '.WEBP', 'webp')):
            img_path = os.path.join(root, file)
            try:
                if img_path.lower().endswith('.webp'):
                    webp_image = imageio.imread(img_path)
                    img = Image.fromarray(webp_image)
                else:
                    img = Image.open(img_path)
                if img is not None:
                    images_pil.append(img)
                    img.close()  # Close the image file
            except Image.UnidentifiedImageError:
                print(f"Skipping non-image file: {img_path}")

print("Loaded", len(images_pil), "images")

def get_data_loader(dataset_path, batch_size, transform):
    dataset = ImageFolder(dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader