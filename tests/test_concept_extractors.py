import unittest

import torch
import torchvision
from torchvision.transforms import transforms

from deep_logic.utils.base import set_seed
from deep_logic.concept_extractor.cnn_models import RESNET18, RESNET101, INCEPTION
from deep_logic.concept_extractor.concept_extractor import CNNConceptExtractor

transform = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.Resize(size=224),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_inception = transforms.Compose([
    transforms.CenterCrop(size=299),
    transforms.Resize(size=299),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
train_data = torchvision.datasets.MNIST(root='../data', train=False,
                                        download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='../data', train=False,
                                       download=True, transform=transform)

train_data_inception = torchvision.datasets.MNIST(root='../data', train=False,
                                                  download=True, transform=transform_inception)
test_data_inception = torchvision.datasets.MNIST(root='../data', train=False,
                                                 download=True, transform=transform_inception)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
epochs = 1

if __name__ == '__main__':
    unittest.main()


class TestConceptExtractor(unittest.TestCase):

    def test_concept_extractor(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor2", cnn_model=RESNET18)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0)
        results.to_csv("results_test_concept_extractor2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 10.
        return

    def test_concept_extractor_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor_pretrained2", cnn_model=RESNET18, pretrained=True)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0)
        results.to_csv("results_test_concept_extractor_pretrained2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 10.
        return

    def test_concept_extractor_big(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor_big2", cnn_model=RESNET101)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0)
        results.to_csv("results_test_concept_extractor_big2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 10.
        return

    def test_concept_extractor_big_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor_big_pretrained2", cnn_model=RESNET101, pretrained=True)
        # It takes a few minutes
        results = model.fit(train_set=train_data, val_set=test_data, epochs=epochs, device=device, num_workers=0)
        results.to_csv("results_test_concept_extractor_big_pretrained2.csv")
        accuracy = model.evaluate(test_data)
        assert accuracy > 10.
        return

    def test_concept_extractor_inception(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor_inception2", cnn_model=INCEPTION)

        # It takes a few minutes
        results = model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                            num_workers=0)
        results.to_csv("results_test_concept_extractor_inception2.csv")
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 10.
        return

    def test_concept_extractor_inception_pretrained(self):
        set_seed(0)
        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.NLLLoss(),
                                    name="test_concept_extractor_inception_pretrained2", cnn_model=INCEPTION,
                                    pretrained=True)

        # It takes a few minutes
        results = model.fit(train_set=train_data_inception, val_set=test_data_inception, epochs=epochs, device=device,
                            num_workers=0)
        results.to_csv("results_test_concept_extractor_inception_pretrained2.csv")
        accuracy = model.evaluate(test_data_inception)
        assert accuracy > 10.
        return
