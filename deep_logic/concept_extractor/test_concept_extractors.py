import unittest

import torch
import torchvision
from torchvision.transforms import transforms

from deep_logic.utils.base import set_seed
from .cnn_models import RESNET18, RESNET101, INCEPTION
from .concept_extractor import CNNConceptExtractor

transform = transforms.Compose([
    transforms.CenterCrop(size=256),
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_inception = transforms.Compose([
    transforms.CenterCrop(size=299),
    transforms.Resize(size=299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
train_data = torchvision.datasets.CIFAR10(root='../data', train=True,
                                          download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False,
                                         download=True, transform=transform)

train_data_inception = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                    download=True, transform=transform_inception)
test_data_inception = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                   download=True, transform=transform_inception)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
epochs = 100
debug = True


if __name__ == '__main__':
    unittest.main()


class TestConceptExtractor(unittest.TestCase):

    def test_concept_extractor(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor2", cnn_model=RESNET18)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, l_r=0.001,
                            device=device, n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor2.csv")
        model.load(device)
        accuracy = model.evaluate(test_data, device=device)
        assert accuracy > 75.
        return

    def test_concept_extractor_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(), transfer_learning=True,
                                    name="test_concept_extractor_pretrained2", cnn_model=RESNET18, pretrained=True)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor_pretrained2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 75.
        return

    def test_concept_extractor_big(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_big2", cnn_model=RESNET101)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor_big2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 75.
        return

    def test_concept_extractor_big_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(), transfer_learning=True,
                                    name="test_concept_extractor_big_pretrained2", cnn_model=RESNET101, pretrained=True)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor_big_pretrained2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 75.
        return

    def test_concept_extractor_inception(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(), transfer_learning=True,
                                    name="test_concept_extractor_inception2", cnn_model=INCEPTION)

        # It takes a few minutes
        results = model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                            n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor_inception2.csv")
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 75.
        return

    def test_concept_extractor_inception_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(), transfer_learning=True,
                                    name="test_concept_extractor_inception_pretrained2", cnn_model=INCEPTION,
                                    pretrained=True)

        # It takes a few minutes
        results = model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                            n_workers=0 if debug else 4)
        results.to_csv("results_test_concept_extractor_inception_pretrained2.csv")
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 75.
        return
