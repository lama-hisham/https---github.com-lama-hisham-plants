import torchvision.models
import torch

def define_model(num_classes):
    model = torchvision.models.alexnet(pretrained=True)
    print("Model loaded:", model)  # Print the model architecture

    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    print("Model modified for", num_classes, "classes")

    return model


#if __name__ == "__main__":
 #   num_classes = 20  
  #  model = define_model(num_classes)
    