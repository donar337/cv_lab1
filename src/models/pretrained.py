import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


def _replace_classifier(module: nn.Module, num_classes: int) -> None:
    if hasattr(module, "fc"):
        in_features = module.fc.in_features
        module.fc = nn.Linear(in_features, num_classes)
    elif hasattr(module, "classifier"):
        if isinstance(module.classifier, nn.Sequential):
            in_features = module.classifier[1].in_features
            module.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            in_features = module.classifier.in_features
            module.classifier = nn.Linear(in_features, num_classes)


def get_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    _replace_classifier(model, num_classes)
    return model


def get_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    _replace_classifier(model, num_classes)
    return model
