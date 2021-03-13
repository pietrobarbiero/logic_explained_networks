import torch
from torch import Tensor
from torch.nn import Linear, Module, Conv2d
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.nn.utils import prune


class XLogicConv2d(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_channels: int, channel_size: int, bias: bool = True, activation: str = 'sigmoid') -> None:
        super(XLogicConv2d, self).__init__(in_channels * channel_size * channel_size, in_channels, bias)
        self.in_channels = in_channels
        self.channel_size = channel_size
        self.activation = activation
        self.top = False
        self.conceptizator = XConceptizator(activation)
        self._prune()

    def forward(self, input: Tensor) -> Tensor:
        input = input.view(-1, self.in_channels * self.channel_size * self.channel_size)
        x = torch.nn.functional.linear(input, self.weight, self.bias)
        # torch.addmm(self.bias, input, self.weight.t())
        # x = torch.matmul(input, self.weight.t()) + self.bias
        x = self.conceptizator(x)
        return x

    def _prune(self):
        # pruning
        blocks = []
        block_size = (1, self.weight.shape[1] // self.in_channels)
        for i in range(self.in_channels):
            blocks.append(torch.ones(block_size))

        mask = torch.block_diag(*blocks)
        prune.custom_from_mask(self, name="weight", mask=mask)
        return

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class XLogic(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, bias: bool = True, activation: str = 'sigmoid', top: bool = False) -> None:
        super(XLogic, self).__init__(in_features, in_features, bias)
        self.in_features = in_features
        self.top = top
        self.activation = activation
        self.conceptizator = XConceptizator(activation)

    def forward(self, input: Tensor) -> Tensor:
        x = torch.nn.functional.linear(input, self.weight, self.bias)
        # torch.addmm(self.bias, input, self.weight.t())
        # x = torch.matmul(input, self.weight.t()) + self.bias
        x = self.conceptizator(x)
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class XRaxor(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """


class XConceptizator(Module):
    """Applies a non-linearity to the incoming data: :math:`y = a(x)`
    """

    def __init__(self, activation: str = 'sigmoid') -> None:
        super(XConceptizator, self).__init__()
        self.concepts = None
        self.activation_name = activation
        self.activation = torch.sigmoid
        if self.activation_name == 'sigmoid':
            self.activation = torch.sigmoid
            self.threshold = 0.5
        if self.activation_name == 'relu':
            self.activation = torch.relu
            self.threshold = 0.
        if self.activation_name == 'leaky_relu':
            self.activation = torch.nn.functional.leaky_relu
            self.threshold = 0.

    def forward(self, input: Tensor) -> Tensor:
        self.concepts = self.activation(input)
        return self.concepts
