import torch
import torchvision.models as models

# Path where the model weights will be saved
save_path = r'C:\Users\Lama\Desktop\disease-detection\https---github.com-lama-hisham-plants\modelweights.pth'

# Initialize the model (ensure this matches your trained model)
num_classes = 20  # Update this to match your number of classes
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)

# Optionally, load the trained weights if you have a trained model
# For demonstration purposes, let's assume you have trained weights and want to load them
# existing_weights_path = r'C:\Users\Lenovo\Desktop\newplantdisease\existing_model_weights.pth'
# model.load_state_dict(torch.load(existing_weights_path))

# Save the model weights
torch.save(model.state_dict(), save_path)
print(f'Model weights saved to {save_path}')