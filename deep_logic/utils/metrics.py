from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score

from .loss import mutual_information


class Metric(ABC):
    """
    Generic metric interface that needs be extended. It always provides the __call__ method.
    """

    @abstractmethod
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Actual calculation of the metric between between computed outputs and given targets.
        :param outputs: predictions
        :param targets: actual labels
        :return: evaluated metric
        """
        pass


class Accuracy(Metric):
    """
    Accuracy computed between the predictions of the model and the actual labels.
    """

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        outputs, targets = outputs.squeeze(), targets.squeeze()
        if len(outputs.shape) > 1:
            # Multi-Label classification
            if len(targets.shape) > 1:
                outputs = outputs > 0.5
            # Multi-Class classification
            else:
                outputs = outputs.argmax(dim=1)
        else:
            # Binary classification
            assert len(targets.shape) == 1, "Target tensor needs to be (N,1) tensor if output is such."
            outputs = outputs > 0.5
        if len(outputs.shape) > 1:
            n_samples = targets.shape[0] * targets.shape[1]
        else:
            n_samples = targets.shape[0]
        accuracy = targets.eq(outputs).sum().item() / n_samples * 100
        return accuracy


class TopkAccuracy(Metric):
    """
    Top-k accuracy computed between the predictions of the model and the actual labels.
    It requires to receive an output tensor of the shape (n,c) where c needs to be greater than 1
    :param k: number of elements of the outputs to consider in order to assert a datum as correctly classified
    """

    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        assert len(outputs.squeeze().shape) > 1, "TopkAccuracy requires a multi-dimensional outputs"
        assert len(targets.squeeze().shape) == 1, "TopkAccuracy requires a single-dimension targets"
        n_samples = targets.shape[0]
        _, topk_outputs = outputs.topk(self.k, 1)
        topk_acc = topk_outputs.eq(targets.reshape(-1, 1)).sum().item() / n_samples * 100
        return topk_acc


class F1Score(Metric):
    """
    F1 score computed on the predictions of a model and the actual labels. Useful for Multi-label classification.
    """

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor, average='macro') -> float:
        outputs = outputs if isinstance(outputs, torch.Tensor) else torch.tensor(outputs)
        targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
        assert len(outputs.squeeze().shape) != 1 or len(targets.squeeze().shape) == 1, \
            "Target tensor needs to be (N,1) tensor if output is such."
        # Multi-class
        if len(outputs.squeeze().shape) > 1 and len(targets.squeeze().shape) == 1:
            discrete_output = outputs.argmax(dim=1)
        # Multi-label or Binary classification
        else:
            discrete_output = outputs.cpu().numpy() > 0.5
        targets = targets.cpu().numpy()
        f1_val = f1_score(discrete_output, targets, average=average, zero_division=0) * 100
        return f1_val


class UnsupervisedMetric(Metric):
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        mi = mutual_information(outputs) * 100

        return mi
