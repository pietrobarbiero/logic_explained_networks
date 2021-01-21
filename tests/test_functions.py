import unittest

import torch


class TestTemplateObject(unittest.TestCase):
    def test_relunets_pruning(self):
        import numpy as np
        from deep_logic.utils.relunn import get_reduced_model
        from deep_logic.utils.base import validate_network, validate_data
        from deep_logic.fol import explain_local, combine_local_explanations
        import deep_logic as dl

        torch.manual_seed(10)
        np.random.seed(0)

        x = torch.tensor([
            [0, 0, 0.5, 0.5],
            [0, 1, 0.5, 0.5],
            [1, 0, 0.5, 0.5],
            [1, 1, 0.5, 0.5],
            [0.5, 0.5, 0, 0],
            [0.5, 0.5, 0, 1],
            [0.5, 0.5, 1, 0],
            [0.5, 0.5, 1, 1],
        ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 0, 2, 3, 3, 3], dtype=torch.long)
        n_classes = len(torch.unique(y))

        layers = [
            torch.nn.Linear(4, 20 * n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(20, 5, n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(5, 1, n_classes),
            torch.nn.Softmax(),
        ]
        model = torch.nn.Sequential(*layers)
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
                dl.utils.relunn.prune_features(model, n_classes)
                need_pruning = False

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = torch.argmax(y_pred, dim=1)
                accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

        explanation = explain_local(model, x, y, x[1], concept_names=['f1', 'f2', 'f3', 'f4'])
        print(explanation)

        for target_class in range(n_classes):
            global_explanation, _, counter = combine_local_explanations(model, x, y, target_class=target_class,
                                                                        topk_explanations=5,
                                                                        concept_names=['f1', 'f2', 'f3', 'f4'])
            print(f'Target class: {target_class} - Explanation: {global_explanation}')
            print(f'\t{counter}')

        return

    def test_relunets(self):
        import numpy as np
        from deep_logic.utils.relunn import get_reduced_model
        from deep_logic.utils.base import validate_network, validate_data
        from deep_logic.fol import explain_local, combine_local_explanations
        import deep_logic as dl

        torch.manual_seed(10)
        np.random.seed(0)

        x = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float)
        y = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        n_classes = len(torch.unique(y))

        layers = [
            torch.nn.Linear(2, 10 * n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(10, 4, n_classes),
            torch.nn.ReLU(),
            dl.nn.XLinear(4, 1, n_classes),
            torch.nn.Sigmoid(),
        ]
        model = torch.nn.Sequential(*layers)
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
                    loss += 0.0001 * torch.norm(module.weight, 1)
            loss.backward()
            optimizer.step()

            if epoch > 500 and need_pruning:
                dl.utils.relunn.prune_features(model, n_classes)
                need_pruning = False

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = torch.argmax(y_pred, dim=1)
                accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

        explanation = explain_local(model, x, y, x[1], concept_names=['f1', 'f2'])
        print(explanation)

        for target_class in range(n_classes):
            global_explanation, _, counter = combine_local_explanations(model, x, y, target_class=target_class,
                                                                        topk_explanations=5,
                                                                        concept_names=['f1', 'f2', 'f3', 'f4'])
            print(f'Target class: {target_class} - Explanation: {global_explanation}')
            print(f'\t{counter}')

        return

    def test_psi_example(self):
        import torch
        import numpy as np
        from deep_logic.utils.base import validate_network
        from deep_logic.utils.sigmoidnn import prune_equal_fanin
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
        from deep_logic.utils.sigmoidnn import prune_equal_fanin, validate_pruning

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        k = 2
        prune_equal_fanin(net, k)
        validate_pruning(net)

        return

    def test_parameter_collection(self):
        from deep_logic.utils.base import collect_parameters

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        weights, biases = collect_parameters(net)

        assert len(weights) == 2
        assert len(biases) == 2

        return

    def test_validation(self):
        from deep_logic.utils.base import validate_data, validate_network

        x = torch.arange(0, 1, step=0.1)
        validate_data(x)

        layers = [torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 1), torch.nn.Sigmoid()]
        net = torch.nn.Sequential(*layers)

        validate_network(net, 'psi')

        return


if __name__ == '__main__':
    unittest.main()
