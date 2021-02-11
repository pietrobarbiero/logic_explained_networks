import torch
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

RESNET10 = "Resnet_10"
RESNET18 = "Resnet_18"
RESNET34 = "Resnet_34"
RESNET50 = "Resnet_50"
RESNET101 = "Resnet_101"
INCEPTION = "Inception"

inception = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)


def ResNet10(pretrained=False):
    assert pretrained == False, "No pretrained weights available for ResNet10"
    return ResNet(BasicBlock, [1, 1, 1, 1])


CNN_MODELS = {
    RESNET10: ResNet10,
    RESNET18: torchvision.models.resnet18,
    RESNET34: torchvision.models.resnet34,
    RESNET50: torchvision.models.resnet50,
    RESNET101: torchvision.models.resnet101,
    INCEPTION: torchvision.models.inception_v3,
}


def get_model(model_name, n_classes, pretrained=True, transfer_learning=True):
    model_dict = CNN_MODELS[model_name]
    model = model_dict(pretrained=pretrained)
    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False
    feat_dim = model.fc.weight.shape[1]
    model.fc = torch.nn.Linear(feat_dim, n_classes)
    if model_name == INCEPTION:
        aux_feat_dim = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = torch.nn.Linear(aux_feat_dim, n_classes)
    return model
