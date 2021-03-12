import unittest

import torch


class TestTemplateObject(unittest.TestCase):
    def test_xor(self):
        from deep_logic.utils.layer import prune_logic_layers
        from deep_logic.logic.layer import explain_class
        import deep_logic as dl

        dl.utils.base.set_seed(0)

        x = torch.tensor([
            [0, 0],#, 0, 0],
            [0, 1],#, 0, 0],
            [0, 1],#, 0, 0],
            [0, 1],#, 0, 0],
            [1, 0],#, 0, 0],
            [1, 1],#, 0, 0],
        ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 1, 1, 0], dtype=torch.float)

        layers = [
            dl.nn.XLogic(2, 4, first=True),
            torch.nn.LeakyReLU(),
            dl.nn.XLogic(4, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.LeakyReLU(),
            dl.nn.XLogic(2, 1, top=True),
            torch.nn.Sigmoid()
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
            # loss = 0

            for module in model.children():
                if isinstance(module, dl.nn.XLogic):
                    # loss += 0.0001 * torch.norm(module.weight, 1)
                    # loss += 0.0001 * torch.norm(module.bias, 1)
                    # break
                    if not module.first and not module.top:
                        cov = 1 / (module.symbols.shape[1] - 1) * torch.matmul(module.symbols.T, module.symbols)
                        cov_np = cov.cpu().detach().numpy()
                        cov_obj = torch.eye(module.symbols.shape[1])
                        loss += 0.001 * torch.norm(cov - cov_obj, p=2)
                        # loss += 1-(module.symbols > 0.5).sum(dim=1).to(float).mean()
                        # cov = 1 / (module.symbols.shape[1] - 1) * torch.matmul(module.symbols.T, module.symbols)
                        # cov_obj = torch.eye(module.symbols.shape[1])
                        # loss += torch.norm(torch.abs(cov - cov_obj), p=1)
                        # break

            loss.backward()
            optimizer.step()

            # if epoch > 1000 and need_pruning:
            #     dl.utils.layer.prune_logic_layers(model)
            #     need_pruning = False

            # compute accuracy
            if epoch % 100 == 0:
                y_pred_d = y_pred > 0.5
                accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

        # cov_np = cov.detach().numpy()

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure()
        sns.heatmap(cov_np)
        plt.show()

        class_explanation, class_explanations = explain_class(model, x, y, target_class=1, simplify=True,
                                                              concept_names=['x0', 'x1', 'x2', 'x3'])

        print(class_explanation)
        print(class_explanations)

        return


if __name__ == '__main__':
    unittest.main()
