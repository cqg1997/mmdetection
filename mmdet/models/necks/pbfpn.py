import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init,build_conv_layer

from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN


@NECKS.register_module()
class PBFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PBFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(PBFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.PBFPN_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            PBFPN_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.PBFPN_convs.append(PBFPN_conv)

        self.downsample_filter = nn.ModuleList()
        self.upsample_filter = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
           # d_f = nn.Conv2d(
           #     out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
           #     groups=out_channels)
           u_f = nn.Conv2d(
               out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
           #  u_f = build_conv_layer(
           #      dict(type='DCN', deform_groups=1),
           #      out_channels,
           #      out_channels,
           #      kernel_size=3,
           #      stride=1,
           #      padding=2,
           #      dilation=2,
           #      bias=True)
            
            # self.downsample_filter.append(d_f)
            self.upsample_filter.append(u_f)

    def init_weights(self):
        super().init_weights()
        for _, filter in enumerate(self.upsample_filter):
            constant_init(filter.conv_offset, 0)
            

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up_feat = F.interpolate(laterals[i], size=prev_shape, mode='nearest')
            
            laterals[i - 1] += self.upsample_filter[i-1](up_feat) 

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            feat = self.downsample_convs[i](inter_outs[i])
            #mask = self.downsample_filter[i](feat)
            inter_outs[i + 1] += feat
        

        outs = [inter_outs[0]]
        outs.extend([
            self.PBFPN_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
