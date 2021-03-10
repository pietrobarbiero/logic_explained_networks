import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F
from torch.nn.utils import prune


class XLogic(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 top: bool = False, first: bool = False) -> None:
        self.top = top
        self.first = first
        super(XLogic, self).__init__(in_features, out_features, bias)
        self.symbols = None
        self.output = None
        self.h = None

    def forward(self, input: Tensor) -> Tensor:
        if self.top:
            self.h = F.linear(input, self.weight, self.bias)
            self.output = torch.sigmoid(self.h)
            self.symbols = self.output
            return self.h
        else:
            if self.first:
                self.symbols = input
            else:
                self.symbols = torch.sigmoid(input)
            self.output = F.linear(self.symbols, self.weight, self.bias)
            return self.output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
