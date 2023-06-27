import torch
import torchvision.models


def resnet_18(model_filepath=None):
    if model_filepath:
        model = torch.load(model_filepath)
    else:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    return model.eval()
