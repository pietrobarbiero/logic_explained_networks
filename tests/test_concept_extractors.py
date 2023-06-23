import unittest

import torch
import torchvision
from torch.utils.data import Subset
from torchvision.transforms import transforms
import numpy as np

from lens.utils.base import set_seed
from lens.models.concept_extractor.cnn_models import RESNET18, RESNET101, INCEPTION
from lens.models.concept_extractor import CNNConceptExtractor

transform = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.Resize(size=224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

transform_inception = transforms.Compose([
    transforms.CenterCrop(size=299),
    transforms.Resize(size=299),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

num_data = 100
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
train_data = torchvision.datasets.MNIST(root='../data', train=True,
                                        download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True, transform=transform)
train_idx = np.random.random_integers(0, len(train_data), num_data)
test_idx = np.random.random_integers(0, len(test_data), num_data)
train_data = Subset(train_data, train_idx)
test_data = Subset(test_data, test_idx)

train_data_inception = torchvision.datasets.MNIST(root='../data', train=True,
                                                  download=True, transform=transform_inception)
test_data_inception = torchvision.datasets.MNIST(root='../data', train=False,
                                                 download=True, transform=transform_inception)
train_data_inception = Subset(train_data_inception, train_idx)
test_data_inception = Subset(test_data_inception, test_idx)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
epochs = 1

if __name__ == '__main__':
    unittest.main()


class TestConceptExtractor(unittest.TestCase):

    def test_concept_extractor_1(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor2", cnn_model=RESNET18)
        # It takes a few minutes
        model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0, save=False)
        accuracy = model.evaluate(test_data)
        assert accuracy > 1.
        return

    def test_concept_extractor_2_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_pretrained2", cnn_model=RESNET18, pretrained=True)
        # It takes a few minutes
        model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0, save=False)
        accuracy = model.evaluate(test_data)
        assert accuracy > 1.
        return

    def test_concept_extractor_3_big(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_big2", cnn_model=RESNET101)
        # It takes a few minutes
        model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0, save=False)
        accuracy = model.evaluate(test_data)
        assert accuracy > 1.
        return

    def test_concept_extractor_4_big_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_big_pretrained2", cnn_model=RESNET101, pretrained=True)
        # It takes a few minutes
        model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0, save=False)
        accuracy = model.evaluate(test_data)
        assert accuracy > 1.
        return

    def test_concept_extractor_5_inception(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_inception2", cnn_model=INCEPTION)

        # It takes a few minutes
        model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                  num_workers=0, save=False)
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 1.
        return

    def test_concept_extractor_6_inception_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss(),
                                    name="test_concept_extractor_inception_pretrained2", cnn_model=INCEPTION,
                                    pretrained=True)

        # It takes a few minutes
        model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                  num_workers=0, save=False)
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 1.
        return
