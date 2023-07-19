# Copyright (c) Facebook, Inc. and its affiliates.
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    get_norm,
)
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import BasicStem, BasicBlock, DeformBottleneckBlock, ResNet


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        return out
