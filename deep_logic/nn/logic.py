import torch
from torch import Tensor
from torch.nn import Linear

from .concepts import XConceptizator


class XLogic(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, activation: str,
                 bias: bool = True, top: bool = False) -> None:
        super(XLogic, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.top = top
        self.conceptizator = XConceptizator(activation)
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:
        x = self.conceptizator(input)
        if not self.top:
            x = torch.nn.functional.linear(x, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class XRaxor(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
