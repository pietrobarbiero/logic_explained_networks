import torch
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

RESNET10 = "Resnet_10"
RESNET18 = "Resnet_18"
RESNET34 = "Resnet_34"
RESNET50 = "Resnet_50"
RESNET101 = "Resnet_101"


def ResNet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


CNN_MODELS = {
    RESNET10: ResNet10,
    RESNET18: ResNet18,
    RESNET34: ResNet34,
    RESNET50: ResNet50,
    RESNET101: RESNET101,
}


def get_model(model_name, n_classes):
    model_dict = CNN_MODELS[model_name]
    resnet = model_dict()
    feat_dim = resnet.fc.weight.shape[1]
    resnet.fc = torch.nn.Linear(feat_dim, n_classes)
    return resnet
