import torchvision.models

def define_model(num_classes):
    model = torchvision.models.alexnet(pretrained=True)
    num_features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_features, num_classes)
    return model