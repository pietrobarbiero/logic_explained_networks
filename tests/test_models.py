import unittest

import torch
from torch.utils.data import TensorDataset
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer

from deep_logic.models.relu_nn import XReluNN
from deep_logic.models.psi_nn import PsiNetwork
from deep_logic.models.general_nn import XGeneralNN
from deep_logic.models.tree import XDecisionTreeClassifier
from deep_logic.models.brl import XBRLClassifier
from deep_logic.utils.base import set_seed
from deep_logic.utils.metrics import Accuracy


# Create data

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float).cpu()
y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).cpu()
y_multi = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float).cpu()
train_data = TensorDataset(x, y)
train_data_multi = TensorDataset(x, y_multi)
x_sample = x[1]
y_sample = y[1]
y_sample_multi = y_multi[1].argmax()

# Define loss and metrics
loss = torch.nn.BCEWithLogitsLoss()
metric = Accuracy()

# Define epochs and learning rate
epochs = 1000
l_r = 0.1

# Network structures
n_features = 2
hidden_neurons = [10, 4]


class TestModels(unittest.TestCase):
    def test_1_relu_nn(self):
        set_seed(0)
        l1_weight_relu = 1e-3

        model = XReluNN(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                        l1_weight=l1_weight_relu)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric, save=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        local_explanation = model.get_local_explanation(x, y, x_sample, target_class=y_sample)
        print(local_explanation)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y, target_class=y_sample)
        print(global_explanation)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'

        # Test with multiple targets
        model = XReluNN(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                        l1_weight=l1_weight_relu)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric, save=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data_multi, metric=metric)
        assert accuracy == 100.0
        print(accuracy)

        local_explanation = model.get_local_explanation(x, y_multi, x_sample, target_class=y_sample_multi)
        print(local_explanation)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y_multi, target_class=y_sample_multi)
        print(global_explanation)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'

        return

    def test_2_psi_nn(self):
        set_seed(0)
        l1_weight_psi = 1e-4

        model = PsiNetwork(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight_psi, fan_in=2)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric, save=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        explanation = model.get_global_explanation(target_class=y_sample)
        print(explanation)
        assert explanation == '((feature0000000000 & ~feature0000000001) | ' \
                              '(feature0000000001 & ~feature0000000000))'

        set_seed(0)
        model = PsiNetwork(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight_psi)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric,
                            save=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data_multi, metric=metric)
        print(accuracy)
        assert accuracy == 100.0

        explanation = model.get_global_explanation(target_class=y_sample_multi)
        print(explanation)
        assert explanation == '((feature0000000000 & ~feature0000000001) | ' \
                              '(feature0000000001 & ~feature0000000000))'
        return

    def test_3_general_nn(self):
        set_seed(0)
        l1_weight_general = 1e-3

        model = XGeneralNN(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight_general)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric, save=False,
                            early_stopping=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        local_explanation = model.get_local_explanation(x, y, x_sample, target_class=y_sample)
        print(local_explanation)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y, target_class=y_sample)
        print(global_explanation)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'

        # Test with multiple targets
        set_seed(0)
        model = XGeneralNN(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight_general)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric, save=False)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data_multi, metric=metric)
        assert accuracy == 100.0
        print(accuracy)

        local_explanation = model.get_local_explanation(x, y_multi, x_sample, target_class=y_sample_multi)
        print(local_explanation)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y_multi, target_class=y_sample_multi)
        print(global_explanation)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'
        return

    def test_4_tree(self):
        set_seed(0)

        model = XDecisionTreeClassifier(n_classes=1, n_features=n_features)

        results = model.fit(train_data, train_data, metric=metric, save=False)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        formula = model.get_global_explanation(class_to_explain=y_sample)
        print(formula)

        model = XDecisionTreeClassifier(n_classes=2, n_features=n_features)

        results = model.fit(train_data_multi, train_data_multi, metric=metric, save=False)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data_multi)

        assert accuracy == 100.0

        formula = model.get_global_explanation(class_to_explain=y_sample_multi)
        print(formula)
        return

    def test_5_brl(self):
        set_seed(0)

        iris = datasets.load_iris()
        x_brl = torch.FloatTensor(iris.data)
        y_brl = torch.FloatTensor(iris.target == 2)
        y_multi_brl = LabelBinarizer().fit_transform(iris.target)
        y_multi_brl = torch.FloatTensor(y_multi_brl)
        train_data_brl = TensorDataset(x_brl, y_brl)
        train_data_multi_brl = TensorDataset(x_brl, y_multi_brl)
        y_sample_multi_brl = y_multi_brl[100].argmax()
        feature_names = iris.feature_names
        class_names = iris.target_names

        model = XBRLClassifier(n_classes=1, n_features=n_features, feature_names=feature_names,
                               class_names=class_names)

        results = model.fit(train_data_brl, train_data_brl, metric=metric, save=False)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data_brl)

        assert accuracy >= 70.0

        formula = model.get_global_explanation(class_to_explain=0)
        print(formula)

        model = XBRLClassifier(n_classes=len(class_names), n_features=n_features, feature_names=feature_names,
                               class_names=class_names)

        results = model.fit(train_data_multi_brl, train_data_multi_brl, metric=metric, save=False)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data_multi_brl)

        assert accuracy >= 70.0

        formula = model.get_global_explanation(class_to_explain=y_sample_multi_brl)
        print(formula)
        return


if __name__ == '__main__':
    unittest.main()
