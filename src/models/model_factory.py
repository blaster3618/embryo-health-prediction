"""
Model Factory – Embryo Health Prediction
=========================================
Supports 12 CNN architectures for model comparison:
  alexnet | vgg16 | vgg19
  resnet18 | resnet50 | resnet101 | resnet152
  densenet121 | densenet201
  inception_v3
  mobilenet_v2 | efficientnet_b0
"""

import torch
import torch.nn as nn
from torchvision import models

# -------------------------------------------------------------------
SUPPORTED_ARCHS = [
    'alexnet', 'vgg16', 'vgg19',
    'resnet18', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet201',
    'inception_v3',
    'mobilenet_v2', 'efficientnet_b0',
]

# Inception v3 requires 299×299; all others use 224×224
ARCH_INPUT_SIZE = {'inception_v3': 299}
DEFAULT_INPUT_SIZE = 224


def get_input_size(arch: str) -> int:
    return ARCH_INPUT_SIZE.get(arch.lower(), DEFAULT_INPUT_SIZE)


def count_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Return a model ready for fine-tuning with the correct output head."""
    arch = arch.lower()
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {SUPPORTED_ARCHS}")

    model = _load_backbone(arch, pretrained)
    _freeze_layers(model, arch)
    _replace_head(model, arch, num_classes)
    return model


def get_gradcam_layer(model: nn.Module, arch: str):
    """Return the last convolutional layer for Grad-CAM."""
    arch = arch.lower()
    mapping = {
        'alexnet':       lambda m: m.features[10],
        'vgg16':         lambda m: m.features[28],
        'vgg19':         lambda m: m.features[34],
        'resnet18':      lambda m: m.layer4[-1].conv2,
        'resnet50':      lambda m: m.layer4[-1].conv3,
        'resnet101':     lambda m: m.layer4[-1].conv3,
        'resnet152':     lambda m: m.layer4[-1].conv3,
        'densenet121':   lambda m: m.features.denseblock4.denselayer16.conv2,
        'densenet201':   lambda m: m.features.denseblock4.denselayer32.conv2,
        'inception_v3':  lambda m: m.Mixed_7c,
        'mobilenet_v2':  lambda m: m.features[-1][0],
        'efficientnet_b0': lambda m: m.features[-1][0],
    }
    if arch not in mapping:
        raise ValueError(f"No Grad-CAM layer defined for '{arch}'")
    return mapping[arch](model)


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _load_backbone(arch: str, pretrained: bool) -> nn.Module:
    W = {
        'alexnet':        models.AlexNet_Weights.DEFAULT,
        'vgg16':          models.VGG16_Weights.DEFAULT,
        'vgg19':          models.VGG19_Weights.DEFAULT,
        'resnet18':       models.ResNet18_Weights.DEFAULT,
        'resnet50':       models.ResNet50_Weights.DEFAULT,
        'resnet101':      models.ResNet101_Weights.DEFAULT,
        'resnet152':      models.ResNet152_Weights.DEFAULT,
        'densenet121':    models.DenseNet121_Weights.DEFAULT,
        'densenet201':    models.DenseNet201_Weights.DEFAULT,
        'inception_v3':   models.Inception_V3_Weights.DEFAULT,
        'mobilenet_v2':   models.MobileNet_V2_Weights.DEFAULT,
        'efficientnet_b0': models.EfficientNet_B0_Weights.DEFAULT,
    }
    loaders = {
        'alexnet':        models.alexnet,
        'vgg16':          models.vgg16,
        'vgg19':          models.vgg19,
        'resnet18':       models.resnet18,
        'resnet50':       models.resnet50,
        'resnet101':      models.resnet101,
        'resnet152':      models.resnet152,
        'densenet121':    models.densenet121,
        'densenet201':    models.densenet201,
        'inception_v3':   models.inception_v3,
        'mobilenet_v2':   models.mobilenet_v2,
        'efficientnet_b0': models.efficientnet_b0,
    }
    weights = W[arch] if pretrained else None
    return loaders[arch](weights=weights)


def _replace_head(model: nn.Module, arch: str, num_classes: int):
    """Replace the final classification layer."""
    if arch == 'alexnet':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif arch in ('vgg16', 'vgg19'):
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif arch in ('resnet18', 'resnet50', 'resnet101', 'resnet152'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch in ('densenet121', 'densenet201'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif arch == 'inception_v3':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(768, num_classes)

    elif arch == 'mobilenet_v2':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif arch == 'efficientnet_b0':
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)


def _freeze_layers(model: nn.Module, arch: str):
    """Freeze early layers; fine-tune deep layers + head."""

    def set_grad(params_iter, requires):
        for p in params_iter:
            p.requires_grad = requires

    if arch == 'resnet18':
        set_grad(model.parameters(), False)
        for sub in [model.layer3, model.layer4, model.fc]:
            set_grad(sub.parameters(), True)

    elif arch in ('resnet50', 'resnet101', 'resnet152'):
        set_grad(model.parameters(), False)
        for sub in [model.layer3, model.layer4, model.fc]:
            set_grad(sub.parameters(), True)

    elif arch in ('vgg16', 'vgg19'):
        set_grad(model.parameters(), False)
        # Fine-tune last conv block (features index ≥20) + classifier
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            elif 'features' in name:
                idx = int(name.split('.')[1])
                param.requires_grad = idx >= 20

    elif arch == 'alexnet':
        set_grad(model.parameters(), False)
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            elif 'features' in name:
                idx = int(name.split('.')[1])
                param.requires_grad = idx >= 8

    elif arch in ('densenet121', 'densenet201'):
        set_grad(model.parameters(), False)
        fine_tune_keys = ['denseblock3', 'denseblock4', 'transition3', 'norm5', 'classifier']
        for name, param in model.named_parameters():
            param.requires_grad = any(k in name for k in fine_tune_keys)

    elif arch == 'inception_v3':
        freeze_prefixes = [
            'Conv2d_1a', 'Conv2d_2a', 'Conv2d_2b', 'Conv2d_3b', 'Conv2d_4a',
            'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a',
        ]
        for name, param in model.named_parameters():
            param.requires_grad = not any(name.startswith(p) for p in freeze_prefixes)

    elif arch == 'mobilenet_v2':
        set_grad(model.parameters(), False)
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            elif 'features' in name:
                idx = int(name.split('.')[1])
                param.requires_grad = idx >= 14

    elif arch == 'efficientnet_b0':
        set_grad(model.parameters(), False)
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            elif 'features' in name:
                idx = int(name.split('.')[1])
                param.requires_grad = idx >= 6
