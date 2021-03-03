from typing import Tuple, List
import collections

import torch
import numpy as np
from sympy import simplify_logic

from .base import replace_names, test_explanation, simplify_formula2
from .psi_nn import _build_truth_table
from ..nn import XLogic
from ..utils.base import collect_parameters, to_categorical
from ..utils.selection import rank_pruning, rank_weights, rank_lime


def explain_class(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  target_class: int, simplify: bool = True, topk_explanations: int = 3,
                  concept_names: List = None) -> str:
    """
    Generate a local explanation for a single sample.

    :param model: pytorch model
    :param x: input samples
    :param y: target labels
    :param x_sample: input for which the explanation is required
    :param target_class: class ID
    :param method: local feature importance method
    :param simplify: simplify local explanation
    :param concept_names: list containing the names of the input concepts
    :param device: cpu or cuda device
    :param num_classes: override the number of classes
    :return: Local explanation
    """
    y = to_categorical(y)
    assert (y == target_class).any(), "Cannot get explanation if target class is not amongst target labels"

    # # collapse samples having the same boolean values and class label different from the target class
    # w, b = collect_parameters(model, device)
    # feature_weights = w[0]
    # feature_used_bool = np.sum(np.abs(feature_weights), axis=0) > 0
    # feature_used = np.sort(np.nonzero(feature_used_bool)[0])
    # _, idx = np.unique((x[:, feature_used][y == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    _, idx = np.unique((x[y == target_class] >= 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    x_target = x[y == target_class][idx]
    y_target = y[y == target_class][idx]
    # x_target = x[y == target_class]
    # y_target = y[y == target_class]
    # print(len(y_target))

    # get model's predictions
    preds = model(x_target)
    preds = to_categorical(preds)

    # identify samples correctly classified of the target class
    correct_mask = y_target.eq(preds)
    x_target_correct = x_target[correct_mask]
    y_target_correct = y_target[correct_mask]

    # collapse samples having the same boolean values and class label different from the target class
    _, idx = np.unique((x[y != target_class] > 0.5).cpu().detach().numpy(), axis=0, return_index=True)
    x_reduced_opposite = x[y != target_class][idx]
    y_reduced_opposite = y[y != target_class][idx]
    preds_opposite = model(x_reduced_opposite)
    if len(preds_opposite.squeeze(-1).shape) > 1:
        preds_opposite = torch.argmax(preds_opposite, dim=1)
    else:
        preds_opposite = (preds_opposite > 0.5).squeeze()

    # identify samples correctly classified of the opposite class
    correct_mask = y_reduced_opposite.eq(preds_opposite)
    x_reduced_opposite_correct = x_reduced_opposite[correct_mask]
    y_reduced_opposite_correct = y_reduced_opposite[correct_mask]

    # select the subset of samples belonging to the target class
    x_validation = torch.cat([x_reduced_opposite_correct, x_target_correct], dim=0)
    y_validation = torch.cat([y_reduced_opposite_correct, y_target_correct], dim=0)

    model.eval()
    model(x_validation)

    concept_names = [f'feature{j:010}' for j in range(x.size(1))]

    class_explanation = ''
    class_explanations = {}
    for layer_id, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, XLogic):
            if module.first:
                prev_module = module

            else:
                explanations = []
                for neuron in range(module.symbols.size(1)):
                    neuron_explanations = []
                    neuron_explanations_raw = {}
                    for i in torch.nonzero(module.symbols[:, neuron] > 0.5):

                        # explanation is the conjunction of non-pruned features
                        explanation_raw = ''
                        for j in torch.nonzero(prev_module.weight.sum(axis=0)):
                            if concept_names[j[0]] not in ['()', '']:
                                if explanation_raw:
                                    explanation_raw += ' & '
                                if prev_module.symbols[i, j[0]] > 0.5:
                                    explanation_raw += concept_names[j[0]]
                                else:
                                    explanation_raw += f'~{concept_names[j[0]]}'

                        # # replace "not True" with "False" and vice versa
                        # explanation = explanation_raw.replace('~(True)', '(False)')
                        # explanation = explanation_raw.replace('~(False)', '(True)')

                        explanation_raw = simplify_logic(explanation_raw, 'dnf', force=True)
                        explanation_raw = str(explanation_raw)
                        if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
                            continue

                        if explanation_raw in neuron_explanations_raw:
                            explanation = neuron_explanations_raw[explanation_raw]
                        elif simplify:
                            explanation = simplify_formula2(explanation_raw, x_validation, y_validation, target_class)
                        else:
                            explanation = explanation_raw

                        if explanation in ['']:
                            continue

                        neuron_explanations_raw[explanation_raw] = explanation
                        neuron_explanations.append(explanation)

                    if len(neuron_explanations) == 0:
                        explanations.append('')
                        class_explanations[f'layer_{layer_id}-neuron_{neuron}'] = ''

                    # get most frequent local explanations
                    counter = collections.Counter(neuron_explanations)
                    topk = topk_explanations
                    if len(counter) < topk_explanations:
                        topk = len(counter)
                    most_common_explanations = []
                    for explanation, _ in counter.most_common(topk):
                        most_common_explanations.append(explanation)

                    # aggregate example-level explanations
                    if neuron_explanations:
                        neuron_explanation = ' | '.join(most_common_explanations)
                        neuron_explanation_simplified = simplify_logic(neuron_explanation, 'dnf', force=False)
                    else:
                        neuron_explanation_simplified = ''
                    explanations.append(f'({neuron_explanation_simplified})')
                    class_explanations[f'layer_{layer_id}-neuron_{neuron}'] = str(neuron_explanation_simplified)

                prev_module = module
                concept_names = explanations
                class_explanation = str(neuron_explanation_simplified)

    return class_explanation, class_explanations
