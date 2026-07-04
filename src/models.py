# src/models.py
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_CLASSES, DEVICE


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256)
        )
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def get_resnet18(pretrained=True, freeze_base=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc    = nn.Linear(in_features, NUM_CLASSES)
    return model


def get_vgg19(pretrained=True, freeze_base=False):
    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.vgg19(weights=weights)

    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features         = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, NUM_CLASSES)
    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_models():
    print("\n TESTING ALL 3 MODELS")

    dummy_input = torch.randn(4, 3, 224, 224)

    models_to_test = [
        ("Custom CNN", CustomCNN()),
        ("ResNet-18",  get_resnet18(pretrained=False)),
        ("VGG-19",     get_vgg19(pretrained=False))
    ]

    for name, model in models_to_test:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        total, trainable = count_parameters(model)
        print(f"\n  {name}")
        print(f"     Input  shape : {list(dummy_input.shape)}")
        print(f"     Output shape : {list(output.shape)}")
        print(f"     Total params      : {total:,}")
        print(f"     Trainable params  : {trainable:,}")

        assert output.shape == (4, NUM_CLASSES), \
            f"Wrong output shape! Expected (4, 14), got {output.shape}"

    print("\n ALL MODELS WORKING! Output shape is correct: [batch, 14]")


if __name__ == "__main__":
    test_models()