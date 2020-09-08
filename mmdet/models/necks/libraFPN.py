import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn.bricks import NonLocal2d
import torch

from ..builder import NECKS


@NECKS.register_module()
class libraFPN(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(libraFPN, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        self.refine_convs_02 = nn.ModuleList()
        self.refine_convs_24 = nn.ModuleList()
        for i in range(3):
            refine_conv = ConvModule(
                in_channels*num_levels,
                in_channels,
                3,
                padding=1,
                groups=in_channels)
            self.refine_convs_02.append(refine_conv)
        for i in range(3):
            refine_conv = ConvModule(
                in_channels*num_levels,
                in_channels,
                3,
                padding=1,
                groups=in_channels)
            self.refine_convs_24.append(refine_conv)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels
        inter_outs = inputs

        # step 1: 0-2nd layers
        refine_level = 1
        # step 1.1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[refine_level].size()[2:]
        sum_size = list(inputs[refine_level].size())
        sum_size[1] *= 3

        for i in range(0, 3):
            if i < refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = torch.cat(feats, dim=2).reshape(sum_size)

        # step 3: scatter refined features to multi-levels by a residual path
        for i in range(0, 3):
            out_size = inputs[i].size()[2:]
            feat = self.refine_convs_02[i](bsf)
            if i < self.refine_level:
                residual = F.interpolate(feat, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(feat, output_size=out_size)
            inter_outs[i] += residual

            # step 1: 0-2nd layers
            refine_level = 3
            # step 1.1: gather multi-level features by resize and average
            feats = []
            gather_size = inputs[refine_level].size()[2:]
            sum_size = list(inputs[refine_level].size())
            sum_size[1] *= 3

            for i in range(2, 5):
                if i < refine_level:
                    gathered = F.adaptive_max_pool2d(
                        inter_outs[i], output_size=gather_size)
                else:
                    gathered = F.interpolate(
                        inter_outs[i], size=gather_size, mode='nearest')
                feats.append(gathered)

            bsf = torch.cat(feats, dim=2).reshape(sum_size)

            # step 3: scatter refined features to multi-levels by a residual path
            for i in range(2, 5):
                out_size = inputs[i].size()[2:]
                feat = self.refine_convs_24[i](bsf)
                if i < self.refine_level:
                    residual = F.interpolate(feat, size=out_size, mode='nearest')
                else:
                    residual = F.adaptive_max_pool2d(feat, output_size=out_size)
                inter_outs[i] += residual

        return tuple(inter_outs)
