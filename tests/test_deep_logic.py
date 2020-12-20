import unittest

import torch


class TestTemplateObject(unittest.TestCase):
    def test_example(self):
        import torch
        import numpy as np
        from deep_logic import validate_network, prune_equal_fanin, collect_parameters
        from deep_logic import fol

        torch.manual_seed(0)
        np.random.seed(0)

        # XOR problem
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1], ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1)

        layers = [torch.nn.Linear(2, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        model = torch.nn.Sequential(*layers)
        validate_network(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for epoch in range(1000):
            # forward pass
            optimizer.zero_grad()
            y_pred = model(x)
            # Compute Loss
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
            # backward pass
            loss.backward()
            optimizer.step()

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = (y_pred > 0.5)
                accuracy = (y_pred_d.eq(y).sum(dim=1) == y.size(1)).sum().item() / y.size(0)
                print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

            # pruning
            if epoch > 500:
                model = prune_equal_fanin(model, 2)

        # generate explanations
        weights, biases = collect_parameters(model)
        f = fol.generate_fol_explanations(weights, biases)[0]
        print(f'Explanation: {f}')

        assert f == '((f1 & ~f2) | (f2 & ~f1))'
        return

    def test_pruning(self):
        from deep_logic import prune_equal_fanin, validate_pruning

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        k = 2
        prune_equal_fanin(net, k)
        validate_pruning(net)

        return

    def test_parameter_collection(self):
        from deep_logic import collect_parameters

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        weights, biases = collect_parameters(net)

        assert len(weights) == 2
        assert len(biases) == 2

        return

    def test_validation(self):
        from deep_logic import validate_data, validate_network

        x = torch.arange(0, 1, step=0.1)
        validate_data(x)

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        validate_network(net)

        return

    def test_fol(self):
        import numpy as np
        from deep_logic.fol import generate_fol_explanations

        w1 = np.array([[1, 0, 2, 0, 0], [1, 0, 3, 0, 0], [0, 1, 0, -1, 0]])
        w2 = np.array([[-1, 0, -2]])
        b1 = [1, 0, -1]
        b2 = [1]

        w = [w1, w2]
        b = [b1, b2]

        f = generate_fol_explanations(w, b)
        assert f[0] == '(f4 | ~f2)'

        return


if __name__ == '__main__':
    unittest.main()
