import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from ..losses import py_bin_sigmoid_focal_loss
from ..builder import HEADS
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class RCHead(BaseDenseHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 region_generator=dict(
                                    type='AnchorGenerator',
                                    octave_base_scale=4 * 2.243,
                                    scales_per_octave=1,
                                    ratios=[1.0],
                                    strides=[8, 16, 32, 64, 128]),
                 train_cfg=None,
                 test_cfg=None):

        super(RCHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.cls_out_channels = num_classes
        self.use_sigmoid_cls = True
        self.sampling = False
        self.background_label = 0
        region_generator = dict(
            type='AnchorGenerator',
            octave_base_scale=4 * 2.243,  # 2e(1/2+2/3)
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128])
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.region_generator = build_anchor_generator(region_generator)
        self.region_assigner = build_assigner(dict(
            type='IoMAssigner',
            pos_iom_thr=0.64,
            neg_iom_thr=0.42,
            min_iom2_thr=0.00,
            ignore_iof_thr=-1))
        self.region_sampler = build_sampler(dict(type='PseudoRegionSampler'), context=self)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers()


    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.region_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.region_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.region_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.region_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.region_cls, std=0.01, bias=bias_cls)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        region_feat = x
        for region_conv in self.region_convs:
            region_feat = region_conv(region_feat)
        # ? ? ? ? ? ? residual structure
        # cls_feat += region_feat
        region_cls = self.region_cls(region_feat)
        hint =  x #+ region_feat
        return region_cls, hint

    def forward(self, feats):
        ret = multi_apply(self.forward_single, feats)
        return ret

    def loss_region_single(self, region_cls, anchors, labels, label_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1, self.cls_out_channels)
        label_weights = label_weights.reshape(-1, self.cls_out_channels)
        region_cls = region_cls.permute(0, 2, 3,
                                       1).reshape(-1, self.cls_out_channels)
        cls_criterion = py_bin_sigmoid_focal_loss
        loss_region_cls = cls_criterion(
            region_cls,
            labels,
            label_weights,
            gamma=2.0,
            alpha=0.25,
            avg_factor=num_total_samples)
        return loss_region_cls[0],None

    def loss(self,
             region_cls,
             holder,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in region_cls]

        assert len(featmap_sizes) == self.region_generator.num_levels
        device = region_cls[0].device
        anchor_list, valid_flag_list = self.get_regions(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_region_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_region_cls, _ = multi_apply(
            self.loss_region_single,
            region_cls,
            all_anchor_list,
            labels_list,
            label_weights_list,
            num_total_samples=num_total_samples)
        loss = dict(loss_region_cls=losses_region_cls)
        return loss

    def get_regions(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.region_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.region_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def _get_region_targets_single(self,
                                   flat_anchors,
                                   valid_flags,
                                   gt_bboxes,
                                   gt_bboxes_ignore,
                                   gt_labels,
                                   img_meta,
                                   label_channels=1,
                                   unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 5
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.region_assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.region_sampler.sample(assign_result, anchors,
                                                     gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        num_gts = gt_labels.shape[0]
        # labels = anchors.new_full((num_gts, label_channels, num_valid_anchors),
        #                           self.background_label,
        #                           dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors, label_channels), dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        labels = []
        if len(pos_inds) > 0:
            for label in  assign_result.gt_inds:
                bin_label = label.new_full((label.size(0), label_channels),0)
                inds = torch.nonzero(label >= 1).squeeze()
                if inds.numel() > 0:
                    bin_label[inds, label[inds] - 1] = 1
                labels.append(bin_label)

            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds, :] = 1.0
            else:
                label_weights[pos_inds, :] = self.train_cfg.pos_weight
            labels = torch.cat(labels,dim=0).reshape(num_gts, num_valid_anchors, label_channels).sum(dim=0)
            labels[labels>0]=1
        else:
            labels = anchors.new_full((num_valid_anchors, label_channels),0, dtype=torch.long)
        if len(neg_inds) > 0:
            label_weights[neg_inds, :] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels,
                num_total_anchors,
                inside_flags,
                fill=self.background_label)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)

        return (labels, label_weights, pos_inds,
                neg_inds, sampling_result)

    def get_region_targets(self,
                           anchor_list,
                           valid_flag_list,
                           gt_bboxes_list,
                           img_metas,
                           gt_bboxes_ignore_list=None,
                           gt_labels_list=None,
                           label_channels=1,
                           unmap_outputs=True,
                           return_sampling_results=False):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_region_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:5]
        rest_results = list(results[5:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        res = (labels_list, label_weights_list,
               num_total_pos, num_total_neg)

        return res

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        raise NotImplementedError
