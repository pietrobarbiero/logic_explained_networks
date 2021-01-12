import unittest

import torch
import numpy as np
import torchvision
from torch import optim
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms

from deep_logic.models.relunn import ReluClassifier
from deep_logic.models.linear import LogisticRegressionClassifier
from deep_logic.models.tree import DecisionTreeClassifier
from deep_logic.utils.base import set_seed, validate_network, validate_data
from deep_logic.utils.metrics import Accuracy, TopkAccuracy
from image_preprocessing.concept_extractor import CNNConceptExtractor


class TestTemplateObject(unittest.TestCase):
    def test_relunn(self):

        set_seed(0)

        # Test with 1 target
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = ReluClassifier(n_classes=1, n_features=2, hidden_neurons=[20, 10, 5], loss=loss, l1_weight=0.001)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.01, metric=metric)
        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data)
        assert accuracy == 100.0

        reduced_model = model.get_reduced_model(x_sample)
        assert isinstance(reduced_model, torch.nn.Sequential)

        explanation = model.get_explanation(x_sample, k=2)
        assert explanation == '~f0 & f1'

        # Test with multiple targets
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        loss = torch.nn.BCELoss()
        metric = TopkAccuracy()
        model = ReluClassifier(n_classes=2, n_features=2, hidden_neurons=[20, 10, 3], loss=loss, l1_weight=0)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.1, metric=metric)
        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data)
        assert accuracy == 100.0

        reduced_model = model.get_reduced_model(x_sample)
        assert isinstance(reduced_model, torch.nn.Sequential)

        explanation = model.get_explanation(x_sample, k=2)
        assert explanation == '~f0 & f1'

        return

    def test_linear(self):

        set_seed(0)

        # Single class test
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 0, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = LogisticRegressionClassifier(n_classes=1, n_features=2, loss=loss)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.01, metric=metric)

        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        # Multi-class test
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = TopkAccuracy()
        model = LogisticRegressionClassifier(n_classes=2, n_features=2, loss=loss)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.01, metric=metric)

        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        return

    def test_tree(self):

        set_seed(0)

        # Single class test
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)

        metric = Accuracy()
        model = DecisionTreeClassifier(n_classes=1, n_features=2)

        results = model.fit(train_data, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        # Multi-class test
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)

        metric = TopkAccuracy()
        model = DecisionTreeClassifier(n_classes=2, n_features=2)

        results = model.fit(train_data, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        return

    def test_concept_extractor(self):
        set_seed(0)

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transform)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        model = CNNConceptExtractor(n_classes=len(classes), loss=torch.nn.CrossEntropyLoss())

        # It takes a few minutes
        results = model.fit(train_set=testset, val_set=testset, epochs=1)

        assert results.shape == (1, 4)

        accuracy = results['Val accs'].values[-1]

        assert accuracy > 25.

        return


if __name__ == '__main__':
    unittest.main()
