import unittest

import torch

import deep_logic as dl
from deep_logic.logic import explain_class
from deep_logic.utils.base import set_seed


class TestTemplateObject(unittest.TestCase):
    def test_explain_class_binary(self):
        for i in range(20):
            set_seed(i)

            # Problem 1
            x = torch.tensor([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

            layers = [
                dl.nn.XLogic(2, activation='identity', first=True),
                torch.nn.Linear(2, 5),
                torch.nn.LeakyReLU(),
                dl.nn.XLogic(5, activation='sigmoid'),
                torch.nn.Linear(5, 1),
                torch.nn.LeakyReLU(),
                dl.nn.XLogic(1, activation='sigmoid', top=True),
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
                            cov = 1 / (module.conceptizator.concepts.shape[1] - 1) * \
                                  torch.matmul(module.conceptizator.concepts.T, module.conceptizator.concepts)
                            cov_np = cov.cpu().detach().numpy()
                            cov_obj = torch.eye(module.conceptizator.concepts.shape[1])
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

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=True,
                                                                  target_class=1,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

        return

    def test_explain_multi_class(self):
        for i in range(20):
            set_seed(i)

            # Problem 1
            x = torch.tensor([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 2], dtype=torch.long)

            layers = [
                dl.nn.XLogic(2, activation='identity', first=True),
                torch.nn.Linear(2, 5),
                torch.nn.LeakyReLU(),
                # dl.nn.XLogic(5, activation='sigmoid'),
                torch.nn.Linear(5, 3),
                torch.nn.LeakyReLU(),
                dl.nn.XLogic(3, activation='identity', top=True),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            need_pruning = True
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y)
                # loss = 0

                for module in model.children():
                    if isinstance(module, dl.nn.XLogic):
                        loss += 0.0001 * torch.norm(module.weight, 1)
                        loss += 0.0001 * torch.norm(module.bias, 1)
                        # break
                        if not module.first and not module.top:
                            cov = 1 / (module.conceptizator.concepts.shape[1] - 1) * \
                                  torch.matmul(module.conceptizator.concepts.T, module.conceptizator.concepts)
                            cov_np = cov.cpu().detach().numpy()
                            cov_obj = torch.eye(module.conceptizator.concepts.shape[1])
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
                    y_pred_d = y_pred.argmax(dim=1)
                    accuracy = y_pred_d.eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=2,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == 'feature0000000000 & feature0000000001'

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=1,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'

            class_explanation, class_explanations = explain_class(model, x, y, binary=False,
                                                                  target_class=0,
                                                                  topk_explanations=10)
            print(class_explanation)
            print(class_explanations)
            assert class_explanation == '~feature0000000000 & ~feature0000000001'

        return


if __name__ == '__main__':
    unittest.main()
