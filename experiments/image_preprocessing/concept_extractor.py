import torch

from models.base import BaseClassifier
from experiments.image_preprocessing.cnn_models import RESNET10, get_model, CNN_MODELS


class CNNConceptExtractor(BaseClassifier):
    """
    CNN classifier used for extracting concepts from images. It follows the strategy employed in Concept Bottleneck
    Models where a classifier (one of the interpretable in our case) is placed on top of a CNN working on images. The
    CNN provides for the low-level concepts, while the classifier provides for the final classification.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param loss: torch.nn.modules.loss
            type of loss to employ
        :param cnn_model: one of the models implemented in the file cnn_models
    """

    def __init__(self, n_classes: int, cnn_model: str = RESNET10,
                 loss: torch.nn.modules.loss = torch.nn.BCELoss(), name: str = "net",
                 device: torch.device = torch.device("cpu")):
        super().__init__(name, device)

        assert cnn_model in CNN_MODELS, f"Required CNN model is not available, it needs to be among {CNN_MODELS.keys()}"

        self.n_classes = n_classes
        self.model = get_model(cnn_model, n_classes)
        self.loss = loss

    def get_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the InterpretableReLU is
        employed. An L1 weight regularization is also always applied

        :param output: output tensor from the forward function
        :param target: label tensor
        :return: loss tensor value
        """
        output_loss = self.loss(output, target)
        return output_loss

    def forward(self, x) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :return: output classification
        """
        super(CNNConceptExtractor, self).forward(x)
        output = self.model(x)
        return output
