from typing import List, Tuple

import torch
from sympy import to_dnf
from sympy.logic.boolalg import is_dnf
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset
from tqdm import trange

from concept_extractor import CNNConceptExtractor
from lens.utils.metrics import Accuracy
from cnn_models import RESNET18


class RobustCNNClassifier(CNNConceptExtractor):

    def __init__(self, n_classes: int, main_classes: List[str], attributes: List[str], cnn_model: str = RESNET18,
                 loss=CrossEntropyLoss(), transfer_learning: bool = False, pretrained: bool = True,
                 name: str = "Robust_CNN", device: torch.device = torch.device("cpu")):
        super().__init__(n_classes, cnn_model, loss, transfer_learning, pretrained, name, device)
        self.main_classes = main_classes
        self._main_class_range = range(len(main_classes))
        self.attributes = attributes
        self.threshold = None
        self.explanations = []
        self._formulas = []
        self.eval_main_classes = False
        self.eval_logits = False

    def forward(self, x, logits=False):
        if self.eval_logits:
            logits = True
        output = super().forward(x, logits)
        if self.eval_main_classes:
            output = output[:, self._main_class_range]
        return output

    def set_eval_main_classes(self, eval: bool = True):
        self.eval_main_classes = eval

    def set_eval_logits(self, eval: bool = True):
        self.eval_logits = eval

    def set_explanations(self, explanations: list) -> None:
        assert len(explanations) == len(self.main_classes), "Error in the explanations of the models. Expected as many " \
                                                            "explanation as the main classes of the model, " \
                                                            f"{len(explanations)} vs {len(self.main_classes)}"
        self.explanations = explanations
        self._calculate_loss_formula()

    def _calculate_loss_formula(self) -> List[List[List[Tuple]]]:
        """
        It is a function which calculates the formulas associated to each class based on the provided explanation.
        The returned formulas will be a list of formulas. Each formula is a list of the terms present in the formula.
        Each term again is a list of attributes present in each term.
        :return: list(list(list(tuple))) formulas as a list of the terms present in each formula represented as a tuple
        (idx: int, negated: bool) where the idx represent the idx of the attribute in the output of the network
        (we must add the number of main classes first)
        """
        self._formulas = []
        for i in trange(len(self.explanations), desc="Setting explanations "):
            explanation = self.explanations[i]
            if not is_dnf(explanation):
                explanation = str(to_dnf(explanation))
            formulas_terms = []
            terms = explanation.split("|")
            assert len(terms) > 0, f"Error in explanation formulation. Expected DNF formula but it is {explanation}"
            for term in terms:
                # excluding trivial explanation
                if term == "False" or term == "True":
                    continue
                # excluding parenthesis
                term = term.replace(" ", "")
                if term[0] == "(":
                    term = term[1:-1]
                formula_term = []
                attributes = term.split("&")
                for attribute in attributes:
                    if attribute[0] == "~":
                        negated = True
                        attribute = attribute[1:]
                    else:
                        negated = False
                    idx = self.attributes.index(attribute)
                    formula_term.append((idx, negated))
                formulas_terms.append(formula_term)
            self._formulas.append(formulas_terms)
        return self._formulas

    def constraint_loss(self, output: torch.Tensor, sum_reduction: bool = True, mu: int = 10, class_disjunction=True) -> torch.Tensor:
        """
        It is a function which calculate the product t-norm losses associated to each formulas. It is expected to
        find in self.formulas the formulas calculated with _calculate_loss_formula().

        Example1:
        class1 -> (attr1 & attr2) | attr3
        => not class1 or ((attr1 & attr2) | attr3)                          (each term in CNF is considered a term)
        => not class1 or (term1 | term2)                                    (converting everything into conjunctions)
        => not class1 or not ((not term1) and (not term2))
        => not ((not (not class1)) and not(not(not term1) and (not term2)))
        => not (class1 and ((not term1) and (not term2)))                   (simplifying)
        => 1 - (class1 * ((1 - term1) * (1 - term2))                        (converting into product t-norm)
        => class1 * (1 - term1) * (1 - term2)                               (converting into loss formulation (1 - formula))
        Example2:
        (attr1 & attr2) | attr3 -> class1
        => not ((attr1 & attr2) | attr3) or class1                          (each term in CNF is considered a term)
        => not (term1 | term2) or class1                                    (converting everything into conjunctions)
        => (not(not((not term1) and (not term2))) or class1                 (not term1 and not term2) = terms
        => (not(not(terms)) or class1
        => terms or class1
        => not(not terms and not class1)                                    (simplifying)
        => 1 - ((1 - terms) * (1 - class1))                                  (converting into product t-norm)
        => (1 - terms)) * (1 - class1)                                      (converting into loss formulation (1 - formula))
        => (1 - ((1 - term1) * (1 - term2)) * (1 - class1)                  (replacing again terms with (1 - term1) * (1 - term2)
        :param output: torch.tensor representing the multi-label output of the network (n_sample) * (n_classes + n_attributes)
        :param sum_reduction: whether to sum the constraint loss over the sample or not (default True)
        :param mu: hyper-parameter factor for balancing the constraint loss
        :param class_disjunction: whether to consider also the disjunction among main classes as logic rules or not
        :return: constraint
        """
        device = self.get_device()
        output = output.to(device)
        assert self._formulas != [], "Explanations not passed yet. set_explanation() method needs to be called first."
        constraint_loss1 = torch.zeros(output.shape[0], device=device)
        constraint_loss2 = torch.zeros(output.shape[0], device=device)
        disjunction_loss = torch.zeros(output.shape[0], device=device)
        for i in range(len(self.main_classes)):
            class_loss1 = output[:, i]
            for term in self._formulas[i]:
                term_loss = 1
                for (attribute, negated) in term:
                    output_attribute = output[:, attribute]
                    if negated:
                        output_attribute = 1 - output_attribute
                    term_loss *= output_attribute
                term_loss = 1 - term_loss
                class_loss1 = class_loss1 * term_loss
            constraint_loss1 += class_loss1
            class_loss2 = 1
            for term in self._formulas[i]:
                term_loss = 1
                for (attribute, negated) in term:
                    output_attribute = output[:, attribute]
                    if negated:
                        output_attribute = 1 - output_attribute
                    term_loss *= output_attribute
                class_loss2 *= (1 - term_loss)
            class_loss2 = (1 - class_loss2) * (1 - output[:, i])
            constraint_loss2 += class_loss2
        if class_disjunction:
            for i in range(len(self.main_classes)):
                single_disjunction_loss = 1
                for j in range(len(self.main_classes)):
                    if i != j:
                        single_disjunction_loss *= (1 - output[:, j])
                disjunction_loss += output[:, i] * (1 - single_disjunction_loss)
        constraint_loss = constraint_loss1 + constraint_loss2 + disjunction_loss
        if sum_reduction:
            return constraint_loss.sum()
        return constraint_loss

    def calc_threshold(self, dataset: Subset, reject_rate: float = 0.1, batch_size=None, metric=Accuracy()):
        n_main_classes = len(self.main_classes)
        with torch.no_grad():
            outputs, labels = self.predict(dataset, device=self.get_device(), batch_size=batch_size)
            output_single_label = outputs[:, :n_main_classes].max(dim=1)[1]
            label_single_label = labels[:, :n_main_classes].max(dim=1)[1]
            acc = metric(output_single_label, label_single_label)
            reject_number = int(reject_rate * len(dataset))
            cons_losses = self.constraint_loss(outputs, sum_reduction=False)
            sorted_cons_losses = cons_losses.sort()[0]
            self.threshold = sorted_cons_losses[len(dataset) - reject_number - 1]
            self.threshold = self.threshold.to(self.get_device())
            rejected = cons_losses > self.threshold
            output_single_label[rejected] = -1
            acc_with_rej = metric(output_single_label, label_single_label)
            print("Threshold", self.threshold.item())
            print("Single label accuracy on val data", acc)
            print("Single label accuracy on val data with rejection", acc_with_rej)

    def evaluate_rejections(self, outputs: torch.Tensor):
        cons_losses = self.constraint_loss(outputs, sum_reduction=False)
        rejected = torch.as_tensor(cons_losses > self.threshold)
        rejection_rate = rejected.sum() / outputs.shape[0]
        return rejection_rate
