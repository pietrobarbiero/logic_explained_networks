import unittest

import torch


class TestTemplateObject(unittest.TestCase):
    def test_xor(self):
        from deep_logic.utils.layer import prune_logic_layers
        from deep_logic.logic.layer import explain_class
        import deep_logic as dl

        dl.utils.base.set_seed(0)

        x = torch.tensor([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.float)

        layers = [
            dl.nn.XLogic(4, 10, first=True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.LeakyReLU(),
            dl.nn.XLogic(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 4),
            torch.nn.LeakyReLU(),
            dl.nn.XLogic(4, 1, top=True),
        ]
        model = torch.nn.Sequential(*layers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.BCELoss()
        model.train()
        need_pruning = True
        for epoch in range(2000):
            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            loss = loss_form(y_pred, y)

            for module in model.children():
                if isinstance(module, dl.nn.XLogic):
                    loss += 0.0001 * torch.norm(module.weight, 1)
                    loss += 0.0001 * torch.norm(module.bias, 1)
                    # break

            loss.backward()
            optimizer.step()

            if epoch > 1000 and need_pruning:
                dl.utils.layer.prune_logic_layers(model)
                need_pruning = False

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = y_pred > 0.5
                accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                print(f'Epoch {epoch}: train accuracy: {accuracy:.4f}')

        class_explanation, class_explanations = explain_class(model, x, y, target_class=1, simplify=True,
                                                              concept_names=['x0', 'x1', 'x2', 'x3'])
        print(class_explanation)
        print(class_explanations)

        return


if __name__ == '__main__':
    unittest.main()
