import torch

from .base import BaseClassifier, BaseXModel
from ..utils.base import NotAvailableError


class XLogisticRegressionClassifier(BaseClassifier, BaseXModel):
    """
        Logistic Regression class module. It does not provides for explanations.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param n_features: int
            number of features - dimension of the input space
        :param loss: torch.nn.modules.loss
            type of loss to employ
     """

    def __init__(self, n_classes: int, n_features: int, loss: torch.nn.modules.loss,
                 device: torch.device = torch.device('cpu'), name: str = "net"):

        super().__init__(loss, name, device)
        self.n_classes = n_classes
        self.n_features = n_features

        layers = [
            torch.nn.Linear(n_features, n_classes),
        ]
        self.model = torch.nn.Sequential(*layers)

    def get_local_explanation(self, x: torch.Tensor, concept_names, **kwargs):
        raise NotAvailableError()

    def get_global_explanation(self, x: torch.Tensor, yclass_to_explain: list, concept_names: list, *args,
                               **kwargs) -> str:
        raise NotAvailableError()

    def prune(self):
        """
        Prune the inputs of the model.

        """
        raise NotAvailableError()


if __name__ == "__main__":
    pass
