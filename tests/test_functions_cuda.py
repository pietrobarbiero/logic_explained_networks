import unittest

import torch

class TestTemplateObject(unittest.TestCase):
    def test_relunets_pruning(self):
        import numpy as np
        from deep_logic.utils.relu_nn import get_reduced_model
        from deep_logic.utils.base import validate_network, validate_data
        from deep_logic.logic import explain_local, explain_global
        import deep_logic as dl

        torch.manual_seed(10)
        np.random.seed(0)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        x = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ], dtype=torch.float).to(device)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.long).to(device)
        n_classes = len(torch.unique(y))

        layers = [
            torch.nn.Linear(4, 20 * n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(20, 5, n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(5, 1, n_classes),
            torch.nn.Softmax(),
        ]
        model = torch.nn.Sequential(*layers).to(device)
        validate_network(model, 'relu')
        validate_data(x)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()
        need_pruning = True
        for epoch in range(1000):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_form(y_pred, y)
            for module in model.children():
                if isinstance(module, torch.nn.Linear):
                    loss += 0.0005 * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()

            if epoch > 500 and need_pruning:
                dl.utils.relu_nn.prune_features(model, n_classes, device)
                need_pruning = False

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = torch.argmax(y_pred, dim=1)
                accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

        explanation = explain_global(model, n_classes, target_class=1,
                                     concept_names=['f1', 'f2', 'f3', 'f4'],
                                     device=device)
        print(explanation)

        explanation = explain_local(model, x, y, x[1], y[1].item(), method='pruning',
                                    concept_names=['f1', 'f2', 'f3', 'f4'], device=device)
        print(explanation)

        for target_class in range(n_classes):
            global_explanation = explain_global(model, n_classes,
                                                target_class=target_class,
                                                concept_names=['f1', 'f2', 'f3', 'f4'],
                                                device=device)
            print(f'Target class: {target_class} - Explanation: {global_explanation}')

        return

    def test_psi_example(self):
        import torch
        import numpy as np
        from deep_logic.utils.base import validate_network
        from deep_logic.utils.psi_nn import prune_equal_fanin
        from deep_logic import logic

        torch.manual_seed(0)
        np.random.seed(0)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # XOR problem
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1], ], dtype=torch.float).to(device)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float).unsqueeze(1).to(device)

        layers = [torch.nn.Linear(2, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        model = torch.nn.Sequential(*layers).to(device)
        validate_network(model, 'psi')

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        need_pruning = True
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
            if epoch > 500 and need_pruning:
                model = prune_equal_fanin(model, 2, device=device)
                need_pruning = False

        # generate explanations
        f = logic.generate_fol_explanations(model, device)[0]
        print(f'Explanation: {f}')

        assert f == '((feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000))'
        return

    def test_pruning(self):
        from deep_logic.utils.psi_nn import prune_equal_fanin, validate_pruning

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers).to(device)

        k = 2
        prune_equal_fanin(net, k, device=device)
        validate_pruning(net)

        return

    def test_parameter_collection(self):
        from deep_logic.utils.base import collect_parameters

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers).to(device)

        weights, biases = collect_parameters(net, device)

        assert len(weights) == 2
        assert len(biases) == 2

        return


if __name__ == '__main__':
    unittest.main()
