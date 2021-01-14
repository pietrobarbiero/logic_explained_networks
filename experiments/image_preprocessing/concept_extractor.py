import torch

from models.base import BaseClassifier
from experiments.image_preprocessing.cnn_models import RESNET18, get_model, CNN_MODELS, INCEPTION


class CNNConceptExtractor(BaseClassifier):
    """
    CNN classifier used for extracting concepts from images. It follows the strategy employed in Concept Bottleneck
    Models where a classifier (one of the interpretable in our case) is placed on top of a CNN working on images. The
    CNN provides for the low-level concepts, while the classifier provides for the final classification.

        :param n_classes: int
            number of classes to classify - dimension of the output layer of the network
        :param loss: torch.nn.modules.loss
            type of loss to employ
        :param cnn_model: str
            one of the models implemented in the file cnn_models
        :param pretrained: bool
            whether to instantiate the model with the weights trained on ImageNet or randomly
    """

    def __init__(self, n_classes: int, class_groups: dict = None, cnn_model: str = RESNET18,
                 loss: torch.nn.modules.loss = torch.nn.BCELoss(), name: str = "net",
                 device: torch.device = torch.device("cpu"), pretrained: bool = True):
        super().__init__(name, device)

        assert cnn_model in CNN_MODELS, f"Required CNN model is not available, it needs to be among {CNN_MODELS.keys()}"
        self.class_groups = class_groups
        self.n_classes = n_classes
        self.cnn_model = cnn_model
        self.model = get_model(cnn_model, n_classes, pretrained=pretrained)
        self.loss = loss
        self._aux_output = None

    def get_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        get_loss method extended from Classifier. The loss passed in the __init__ function of the InterpretableReLU is
        employed. An L1 weight regularization is also always applied

        :param outputs: output tensor from the forward function
        :param targets: label tensor
        :return: loss tensor value
        """
        loss = self.loss(outputs, targets)
        if self.cnn_model == INCEPTION and self._aux_output is not None:
            loss += 0.4 * self.loss(self._aux_output, targets)
        return loss

    def forward(self, x, logits=False) -> torch.Tensor:
        """
        forward method extended from Classifier. Here input data goes through the layer of the ReLU network.
        A probability value is returned in output after sigmoid activation

        :param x: input tensor
        :param logits: whether to return logits activation or sigmoid output
        :return: output classification
        """
        super(CNNConceptExtractor, self).forward(x)
        output = self.model(x)

        # Inception return 2 logits tensor
        if self.cnn_model == INCEPTION:
            self._aux_output = output[1] if logits else torch.sigmoid(output[1])
            output = output[0]

        if logits:
            return output
        return torch.sigmoid(output)
