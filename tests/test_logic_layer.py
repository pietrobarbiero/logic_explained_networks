import unittest

import torch

import deep_logic as dl
from deep_logic.logic import explain_class
from deep_logic.utils.base import set_seed
from deep_logic.utils.layer import prune_logic_layers


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
                dl.nn.XLogic(2, 5, first=True),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 1),
                dl.nn.XLogic(1, 1, top=True),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.BCELoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x).squeeze()
                loss = loss_form(y_pred, y)

                loss.backward()
                optimizer.step()

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

    def test_explain_class_binary_pruning(self):
        for i in range(20):
            set_seed(i)

            # Problem 1
            x = torch.tensor([
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 0, 1],
            ], dtype=torch.float)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

            layers = [
                dl.nn.XLogic(4, 5, first=True),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 1),
                dl.nn.XLogic(1, 1, top=True),
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
                        break

                loss.backward()
                optimizer.step()

                if epoch > 1000 and need_pruning:
                    prune_logic_layers(model, fan_in=2)
                    need_pruning = False

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
                dl.nn.XLogic(2, 5, first=True),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 5),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(5, 3),
                dl.nn.XLogic(3, 3, activation='identity', top=True),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(2000):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_form(y_pred, y)

                loss.backward()
                optimizer.step()

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
