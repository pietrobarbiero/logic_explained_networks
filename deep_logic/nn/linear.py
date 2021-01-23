import torch
from torch.nn import Linear
from torch.nn.utils import prune


class XLinear(Linear):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """

    def __init__(self, in_features: int, out_features: int, n_classes: int, bias: bool = True) -> None:
        self.n_classes = n_classes
        super(XLinear, self).__init__(n_classes * in_features, n_classes * out_features, bias)

        # pruning
        blocks = []
        block_size = (self.weight.shape[0] // self.n_classes, self.weight.shape[1] // self.n_classes)
        for i in range(self.n_classes):
            blocks.append(torch.ones(block_size))

        mask = torch.block_diag(*blocks)
        prune.custom_from_mask(self, name="weight", mask=mask)
