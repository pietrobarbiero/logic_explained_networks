import unittest

import torch
import numpy as np
from sklearn.metrics import accuracy_score

from deep_logic.logic import test_explanation
from deep_logic.logic.base import simplify_formula, simplify_formula

features = [f"feature{i:010}" for i in range(10)]

class TestTemplateObject(unittest.TestCase):
    def test_test_explanation_binary(self):
        # Problem 1
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=torch.float)
        y = torch.tensor([0, 0, 0, 1], dtype=torch.float)

        # Test 1
        explanation = f'~{features[0]} | ~{features[1]})'
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [True, True, True, False])

        explanation = f'{features[0]} & {features[1]}'
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, False, False, True])

        accuracy, preds = test_explanation(explanation,
                                           target_class=1, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, False, False, True])

        # Problem 2
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 0], dtype=torch.float)

        # Test 1
        explanation = f'(~{features[0]} & ~{features[1]}) | ({features[0]} & {features[1]})'
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [True, False, False, True])

        explanation = f'({features[0]} & ~{features[1]}) | (~{features[0]} & {features[1]})'
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, True, True, False])

        accuracy, preds = test_explanation(explanation,
                                           target_class=1, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, True, True, False])

        return

    def test_test_explanation_multi_class(self):
        # Problem 1
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=torch.float)
        y = torch.tensor([0, 2, 2, 1], dtype=torch.float)

        # Test 1
        explanation = f'~{features[0]} & ~{features[1]}'
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [True, False, False, False])

        explanation = f'{features[0]} & {features[1]}'
        accuracy, preds = test_explanation(explanation,
                                           target_class=1, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, False, False, True])

        explanation = f'({features[0]} & ~{features[1]}) | (~{features[0]} & {features[1]})'
        accuracy, preds = test_explanation(explanation,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, True, True, False])

        explanation = f'~{features[0]} & {features[1]}'
        accuracy, preds = test_explanation(explanation,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} \n\t '
              f'Accuracy: {accuracy:.2f} \n\t '
              f'Predictions: {preds}')
        assert np.all(preds == [False, True, False, False])

        return

    def test_test_simplify_formula_binary(self):
        # Problem 1
        x = torch.tensor([
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
        ], dtype=torch.float)
        y = torch.tensor([0, 1, 1, 1], dtype=torch.float)

        # Test 1
        explanation = f'~{features[0]} & ~{features[1]} & {features[2]} & ~{features[3]}'
        explanation2 = simplify_formula(explanation, x=x, y=y, target_class=0)
        accuracy, preds = test_explanation(explanation,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        accuracy2, preds2 = test_explanation(explanation2,
                                           target_class=0, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} VS Formula2: {explanation2} \n\t '
              f'Accuracy: {accuracy:.2f} VS Accuracy2: {accuracy2:.2f} \n\t '
              f'Predictions: {preds} VS Predictions2: {preds2}')
        assert np.all(preds == preds2)

        return

    def test_test_simplify_formula_multi_class(self):
        # Problem 1
        x = torch.tensor([
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
        ], dtype=torch.float)
        y = torch.tensor([0, 2, 1, 1], dtype=torch.float)

        # Test 1
        explanation = f'~{features[0]} & {features[1]} & {features[2]} & ~{features[3]}'
        explanation2 = simplify_formula(explanation, x=x, y=y, target_class=2)
        accuracy, preds = test_explanation(explanation,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        accuracy2, preds2 = test_explanation(explanation2,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} VS Formula2: {explanation2} \n\t '
              f'Accuracy: {accuracy:.2f} VS Accuracy2: {accuracy2:.2f} \n\t '
              f'Predictions: {preds} VS Predictions2: {preds2}')
        assert np.all(preds == preds2)

        # Test 2
        explanation = f'~{features[0]} & ~{features[1]} & {features[2]} & ~{features[3]}'
        explanation2 = simplify_formula(explanation, x=x, y=y, target_class=2)
        accuracy, preds = test_explanation(explanation,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        accuracy2, preds2 = test_explanation(explanation2,
                                           target_class=2, x=x, y=y,
                                           metric=accuracy_score)
        print(f'Formula: {explanation} VS Formula2: {explanation2} \n\t '
              f'Accuracy: {accuracy:.2f} VS Accuracy2: {accuracy2:.2f} \n\t '
              f'Predictions: {preds} VS Predictions2: {preds2}')
        assert np.all(preds == preds2)

        return


if __name__ == '__main__':
    unittest.main()
