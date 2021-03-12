import torch
from torch import Tensor
from torch.nn import Linear, Parameter, Module
from torch.nn import functional as F
from torch.nn.utils import prune


class XLogic(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = 'sigmoid', top: bool = False, first: bool = False) -> None:
        self.top = top
        self.first = first
        super(XLogic, self).__init__(in_features, out_features, bias)
        self.symbols = None
        self.output = None
        self.h = None
        self.activation_name = activation
        self.activation = F.sigmoid
        if self.activation_name == 'sigmoid':
            self.activation = F.sigmoid
        if self.activation_name == 'relu':
            self.activation = F.relu
        if self.activation_name == 'leaky_relu':
            self.activation = F.leaky_relu
        # self.whitening_matrix = Parameter(torch.Tensor(in_features, in_features))
        # self.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        if self.top:
            self.h = F.linear(input, self.weight, self.bias)
            self.output = self.activation(self.h)
            self.symbols = self.output
            return self.h
        else:
            if self.first:
                self.symbols = input
            else:
                self.symbols = self.activation(input)
            self.output = F.linear(self.symbols, self.weight, self.bias)
            return self.output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class XRaxor(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """


class XConceptizator(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, activation: str = 'sigmoid') -> None:
        super(XConceptizator, self).__init__()
        self.concepts = None
        self.activation_name = activation
        self.activation = F.sigmoid
        if self.activation_name == 'sigmoid':
            self.activation = F.sigmoid
        if self.activation_name == 'relu':
            self.activation = F.relu
        if self.activation_name == 'leaky_relu':
            self.activation = F.leaky_relu

    def forward(self, input: Tensor) -> Tensor:
        self.concepts = self.activation(input)
        return self.concepts
