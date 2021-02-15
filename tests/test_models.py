import unittest

import torch
from torch.utils.data import TensorDataset

from deep_logic.models.relu_nn import XReluNN
from deep_logic.models.psi_nn import PsiNetwork
from deep_logic.models.tree import XDecisionTreeClassifier
from deep_logic.utils.base import set_seed
from deep_logic.utils.metrics import Accuracy
from deep_logic.models.general_nn import XGeneralNN


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
loss = torch.nn.BCELoss()
metric = Accuracy()

# Define epochs and learning rate
epochs = 1000
l_r = 0.01
l1_weight = 1e-5

# Network structures
n_features = 2
hidden_neurons = [4]


class TestModels(unittest.TestCase):
    def test_1_relu_nn(self):
        set_seed(0)

        model = XReluNN(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                        l1_weight=l1_weight)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        local_explanation = model.get_local_explanation(x, y, x_sample, target_class=y_sample)
        assert local_explanation == '~feature0000000000'

        global_explanation = model.get_global_explanation(x, y, target_class=y_sample)
        assert global_explanation == '~feature0000000000 | ~feature0000000001'

        # Test with multiple targets
        model = XReluNN(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                        l1_weight=l1_weight)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data_multi, metric=metric)
        assert accuracy == 100.0
        print(accuracy)

        local_explanation = model.get_local_explanation(x, y_multi, x_sample, target_class=y_sample_multi)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y_multi, target_class=y_sample_multi)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'

        return

    def test_2_psi_nn(self):
        set_seed(0)

        model = PsiNetwork(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight, fan_in=2)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        explanation = model.get_global_explanation(target_class=y_sample)
        assert explanation == '((feature0000000000 & ~feature0000000001) | ' \
                              '(feature0000000001 & ~feature0000000000))'

        model = PsiNetwork(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric)
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

        model = XGeneralNN(n_classes=1, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight)

        results = model.fit(train_data, train_data, epochs=epochs, l_r=l_r, metric=metric)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data, metric=metric)
        assert accuracy == 100.0

        local_explanation = model.get_local_explanation(x, y, x_sample, target_class=y_sample)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y, target_class=y_sample)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'

        # Test with multiple targets
        model = XGeneralNN(n_classes=2, n_features=n_features, hidden_neurons=hidden_neurons, loss=loss,
                           l1_weight=l1_weight)

        results = model.fit(train_data_multi, train_data_multi, epochs=epochs, l_r=l_r, metric=metric)
        assert results.shape == (epochs, 4)

        accuracy = model.evaluate(train_data_multi, metric=metric)
        assert accuracy == 100.0
        print(accuracy)

        local_explanation = model.get_local_explanation(x, y_multi, x_sample, target_class=y_sample_multi)
        assert local_explanation == '~feature0000000000 & feature0000000001'

        global_explanation = model.get_global_explanation(x, y_multi, target_class=y_sample_multi)
        assert global_explanation == '(feature0000000000 & ~feature0000000001) | ' \
                                     '(feature0000000001 & ~feature0000000000)'
        return

    def test_4_tree(self):
        set_seed(0)

        model = XDecisionTreeClassifier(n_classes=1, n_features=n_features)

        results = model.fit(train_data, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data)

        assert accuracy == 100.0

        formula = model.get_global_explanation(class_to_explain=y_sample)
        print(formula)

        model = XDecisionTreeClassifier(n_classes=2, n_features=n_features)

        results = model.fit(train_data_multi, train_data, metric=metric)

        assert results.shape == (1, 4)

        accuracy = model.evaluate(train_data_multi)

        assert accuracy == 100.0

        formula = model.get_global_explanation(class_to_explain=y_sample_multi)
        print(formula)
        return

#
# if __name__ == '__main__':
#     unittest.main()
