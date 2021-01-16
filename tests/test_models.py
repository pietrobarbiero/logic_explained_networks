import os
import unittest

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms

from deep_logic.models.linear import XLogisticRegressionClassifier
from deep_logic.models.relunn import XReluClassifier
from deep_logic.models.sigmoidnn import XSigmoidClassifier
from deep_logic.models.tree import XDecisionTreeClassifier
from deep_logic.utils.base import set_seed
from deep_logic.utils.metrics import Accuracy, TopkAccuracy
from image_preprocessing.cnn_models import RESNET18, RESNET50, INCEPTION, RESNET101
from image_preprocessing.concept_extractor import CNNConceptExtractor


class TestModels(unittest.TestCase):
    def test_relunn(self):
        set_seed(0)

        # Test with 1 target
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XReluClassifier(n_classes=1, n_features=2, hidden_neurons=[20, 10, 5], loss=loss, l1_weight=0.001)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.01, metric=metric)
        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        reduced_model = model.get_reduced_model(x_sample)
        assert isinstance(reduced_model, torch.nn.Sequential)

        explanation = model.explain(x, y, sample_id=0, local=True)
        assert explanation == 'feature00001 & ~feature00000'

        # Test with multiple targets
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XReluClassifier(n_classes=2, n_features=2, hidden_neurons=[20, 10, 3], loss=loss, l1_weight=0.001)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.01, metric=metric)
        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0
        print(accuracy)

        reduced_model = model.get_reduced_model(x_sample)
        assert isinstance(reduced_model, torch.nn.Sequential)

        # TODO: not implemented yet
        # explanation = model.explain(x, y, sample_id=0, local=True)
        # assert explanation == 'feature00001 & ~feature00000'
        # return

    def test_sigmoidnn(self):
        set_seed(0)

        # Test with 1 target
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XSigmoidClassifier(n_classes=1, n_features=2, hidden_neurons=[5, 3], loss=loss, l1_weight=0.001)

        results = model.fit(train_data, train_data, batch_size=4, epochs=1000, l_r=0.01, metric=metric)
        assert results.shape == (1000, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        print(accuracy)
        assert accuracy == 100.0

        explanation = model.explain()
        print(explanation)
        assert explanation == ['(f1)']

        # Test with multiple targets
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XSigmoidClassifier(n_classes=2, n_features=2, hidden_neurons=[20, 10, 3], loss=loss, l1_weight=0)

        results = model.fit(train_data, train_data, batch_size=4, epochs=1000, l_r=0.01, metric=metric)
        assert results.shape == (1000, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        print(accuracy)
        assert accuracy == 100.0

        explanation = model.explain()
        print(explanation)
        assert explanation == ['(f2)', '(~f2)']

        return

    def test_linear(self):
        set_seed(0)

        # Single class test
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        # For the linear classifier we cannot use the real XOR problem since it cannot accomplish it
        y = torch.tensor([0, 1, 0, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XLogisticRegressionClassifier(n_classes=1, n_features=2, loss=loss)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.1, metric=metric)

        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        # Multi-class test
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        # For the linear classifier we cannot use the real XOR problem since it cannot accomplish it
        y = torch.tensor([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)

        loss = torch.nn.BCELoss()
        metric = Accuracy()
        model = XLogisticRegressionClassifier(n_classes=2, n_features=2, loss=loss)

        results = model.fit(train_data, train_data, batch_size=4, epochs=100, l_r=0.1, metric=metric)

        assert results.shape == (100, 4)

        accuracy = model.evaluate(train_data, metric=metric)

        assert accuracy == 100.0

        return

    def test_tree(self):
        set_seed(0)

        # Single class test
        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float).cpu()
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
        train_data = TensorDataset(x, y)

        metric = Accuracy()
        model = XDecisionTreeClassifier(n_classes=1, n_features=2)

        results = model.fit(train_data, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        # Multi-class test
        x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
        y = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float).cpu()
        train_data = TensorDataset(x, y)

        metric = Accuracy()
        model = XDecisionTreeClassifier(n_classes=2, n_features=2)

        results = model.fit(train_data, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0
        return


if __name__ == '__main__':
    unittest.main()
