from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score


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
        if len(outputs.squeeze().shape) > 1:
            outputs = outputs.argmax(dim=1)
        if len(targets.squeeze().shape) > 1:
            targets = targets.argmax(dim=1)
        n_samples = targets.shape[0]
        accuracy = targets.eq(outputs>0.5).sum().item() / n_samples * 100
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
        assert len(outputs.squeeze().shape) > 1, "TopkAccuracy requires a multi-dimensional outputs"
        if len(targets.squeeze().shape) > 1:
            targets = targets.argmax(dim=1)
        n_samples = targets.shape[0]
        _, topk_outputs = outputs.topk(self.k, 1)
        topk_acc = topk_outputs.eq(targets.reshape(-1, 1)).sum().item() / n_samples * 100
        return topk_acc


class F1Score(Metric):
    """
    F1 score computed on the predictions of a model and the actual labels. Useful for Multi-label classification.
    """

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        discrete_output = outputs.cpu().numpy() > 0.5
        f1_val = f1_score(discrete_output, targets.cpu().numpy(), average='macro', zero_division=0) * 100
        return f1_val
