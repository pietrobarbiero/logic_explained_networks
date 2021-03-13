from typing import Tuple, List
import collections

import torch
import numpy as np
from sympy import simplify_logic

from .base import replace_names, test_explanation, simplify_formula2
from .psi_nn import _build_truth_table
from ..nn import XLogic, XLogicConv2d
from ..utils.base import collect_parameters, to_categorical
from ..utils.selection import rank_pruning, rank_weights, rank_lime


def explain_class(model: torch.nn.Module, x: torch.Tensor, concepts_in: torch.Tensor,
                  y: torch.Tensor,
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
    threshold = 0.
    _, idx = np.unique((concepts_in[y == target_class] >= threshold).cpu().detach().numpy(), axis=0, return_index=True)
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
    _, idx = np.unique((concepts_in[y != target_class] > threshold).cpu().detach().numpy(), axis=0, return_index=True)
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

    model.train()
    model(x_validation)

    class_explanation = ''
    class_explanations = {}
    is_first = True
    for layer_id, module in enumerate(model.children()):
        # prune only Linear layers
        if isinstance(module, XLogic) or isinstance(module, XLogicConv2d):

            if is_first:
                prev_module = module
                is_first = False
                feature_names = [f'feature{j:010}' for j in range(prev_module.conceptizator.concepts.size(1))]
                c_validation = prev_module.conceptizator.concepts

            else:
                explanations = []
                for neuron in range(module.conceptizator.concepts.size(1)):
                    neuron_explanations = []
                    neuron_explanations_raw = {}
                    for i in torch.nonzero(module.conceptizator.concepts[:, neuron] > module.conceptizator.threshold):

                        # explanation is the conjunction of non-pruned features
                        explanation_raw = ''
                        for j in torch.nonzero(prev_module.weight.sum(axis=1)):
                            if feature_names[j[0]] not in ['()', '']:
                                if explanation_raw and prev_module.conceptizator.concepts[i, j[0]] > module.conceptizator.threshold:
                                    explanation_raw += ' & '
                                if prev_module.conceptizator.concepts[i, j[0]] > module.conceptizator.threshold:
                                    explanation_raw += feature_names[j[0]]
                                # if explanation_raw:
                                #     explanation_raw += ' & '
                                # if prev_module.conceptizator.concepts[i, j[0]] > module.conceptizator.threshold:
                                #     explanation_raw += feature_names[j[0]]
                                # else:
                                #     explanation_raw += f'~{feature_names[j[0]]}'

                        # # replace "not True" with "False" and vice versa
                        # explanation = explanation_raw.replace('~(True)', '(False)')
                        # explanation = explanation_raw.replace('~(False)', '(True)')

                        if explanation_raw:
                            explanation_raw = simplify_logic(explanation_raw, 'dnf', force=True)
                        explanation_raw = str(explanation_raw)
                        if explanation_raw in ['', 'False', 'True', '(False)', '(True)']:
                            continue

                        if explanation_raw in neuron_explanations_raw:
                            explanation = neuron_explanations_raw[explanation_raw]
                        elif simplify:
                            explanation = simplify_formula2(explanation_raw, c_validation, y_validation, target_class)
                        else:
                            explanation = explanation_raw

                        if explanation in ['']:
                            continue

                        neuron_explanations_raw[explanation_raw] = explanation
                        neuron_explanations.append(explanation)

                    if len(neuron_explanations) == 0:
                        explanations.append('')
                        class_explanations[f'layer_{layer_id}-neuron_{neuron}'] = ''

                    else:
                        # get most frequent local explanations
                        counter = collections.Counter(neuron_explanations)
                        topk = topk_explanations
                        if len(counter) < topk_explanations:
                            topk = len(counter)
                        most_common_explanations = []
                        for explanation, _ in counter.most_common(topk):
                            most_common_explanations.append(f'({explanation})')

                        # aggregate example-level explanations
                        if neuron_explanations:
                            neuron_explanation = ' | '.join(most_common_explanations)
                            neuron_explanation_simplified = simplify_logic(neuron_explanation, 'dnf', force=False)
                        else:
                            neuron_explanation_simplified = ''
                        explanations.append(f'({neuron_explanation_simplified})')
                        class_explanations[f'layer_{layer_id}-neuron_{neuron}'] = str(neuron_explanation_simplified)

                prev_module = module
                feature_names = explanations
                class_explanation = str(neuron_explanation_simplified)

    # replace concept names
    if concept_names is not None:
        class_explanation = replace_names(class_explanation, concept_names)
        for k, explanation in class_explanations.items():
            class_explanations[k] = replace_names(explanation, concept_names)

    return class_explanation, class_explanations
