import torch
from torch.nn import Linear
from torch.nn.utils import prune


class XLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        n_classes: number of classes
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
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
