import unittest

import torch


class TestTemplateObject(unittest.TestCase):
    def test_relunets(self):
        import numpy as np
        from deep_logic import get_reduced_model, validate_network, validate_data
        from deep_logic.fol import generate_local_explanations, combine_local_explanations

        torch.manual_seed(10)
        np.random.seed(0)

        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float)
        y = torch.tensor([1, 0, 1, 0], dtype=torch.float).unsqueeze(1)
        x_sample = torch.tensor([0, 1], dtype=torch.float)

        layers = [
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid(),
        ]
        model = torch.nn.Sequential(*layers)
        validate_network(model, 'relu')
        validate_data(x)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for epoch in range(1000):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y)
            for module in model.children():
                if isinstance(module, torch.nn.Linear):
                    loss += 0.0001 * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()
        explanation = combine_local_explanations(model, x, y)
        print(explanation)

        model.eval()
        y_pred = model(x_sample)

        model_reduced = get_reduced_model(model, x_sample)
        y_pred_reduced = model_reduced(x_sample)

        explanation = generate_local_explanations(model_reduced, x_sample)
        print(explanation)

        assert y_pred.eq(y_pred_reduced)[0]

        return

    def test_psi_example(self):
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
        validate_network(model, 'psi')

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
        f = fol.generate_fol_explanations(model)[0]
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

        validate_network(net, 'psi')

        return


if __name__ == '__main__':
    unittest.main()
