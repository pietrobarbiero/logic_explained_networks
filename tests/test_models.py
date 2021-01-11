import unittest

import torch
import numpy as np
from torch.utils.data import TensorDataset

from deep_logic.models.relunn import XReluClassifier
from deep_logic.utils.base import set_seed, validate_network, validate_data


class TestTemplateObject(unittest.TestCase):
    def test_relunets(self):

        set_seed(0)

        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float)
        y = torch.tensor([1, 0, 1, 0], dtype=torch.float).unsqueeze(1)
        train_data = TensorDataset(x, y)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        loss = torch.nn.BCELoss()
        model = XReluClassifier(n_classes=1, n_features=2, hidden_neurons=[20, 10, 5], loss=loss)

        model.fit(train_data, train_data, batch_size=4, epochs=1000, l_r=0.01)

        return


if __name__ == '__main__':
    unittest.main()
