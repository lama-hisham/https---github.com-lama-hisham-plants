import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.Image as Image

def get_preprocessed_transform():
    print("Entering get_preprocessed_transform function")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print("Returning transform from get_preprocessed_transform function")
    return transform
''''
# Load an example image
img = Image.open(r'C:\Users\Lama\Downloads\extractedcitrus\Citrus\Aphids\1(1).jpg')

# Display the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# Apply the transformation to the image
transform = get_preprocessed_transform()
img_transformed = transform(img)

# Convert the transformed image back to PIL image for display
img_transformed = img_transformed.numpy().transpose((1, 2, 0))
img_transformed = Image.fromarray((img_transformed * 255).astype('uint8'))

# Display the transformed image
plt.subplot(1, 2, 2)
plt.imshow(img_transformed)
plt.title('Transformed Image')
plt.axis('off')
plt.show()

'''