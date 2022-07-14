# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..relation_head.utils_motifs import obj_edge_vectors
from maskrcnn_benchmark.data import get_dataset_statistics
from ..relation_head.model_motifs import FrequencyBias


@registry.ROI_WINDOW_RELATION_FEATURE_EXTRACTORS.register("WindowRelationFeatureExtractor")
class WindowRelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(WindowRelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
        self.out_channels = self.feature_extractor.out_channels

        # use the whole window to predict the scores
        self.use_window_box = self.cfg.MODEL.ROI_WINDOW_HEAD.PREDICT_USE_WINDOW
        if self.use_window_box:
            self.win_cls = make_fc(self.out_channels, self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)

        # use the provided pairs to predict the scores for the whole windows
        self.use_pairs = self.cfg.MODEL.ROI_WINDOW_HEAD.PREDICT_USE_PAIRS
        if self.use_pairs:
            self.use_semantics = self.cfg.MODEL.ROI_WINDOW_HEAD.PREDICT_USE_SEMANTICS
            if self.use_semantics:
                statistics = get_dataset_statistics(self.cfg)
                self.freq_bias = FrequencyBias(self.cfg, statistics)

            # separete spatial
            self.spatial_on = self.cfg.MODEL.ROI_WINDOW_HEAD.SPATIAL_ON
            self.separate_spatial = self.cfg.MODEL.ROI_WINDOW_HEAD.SEPARATE_SPATIAL
            if self.spatial_on:
                if self.separate_spatial:
                    input_size = self.feature_extractor.resize_channels
                    out_dim = self.feature_extractor.out_channels
                    self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                                      make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                                    ])

                # union rectangle size
                self.rect_size = resolution * 4 -1
                self.rect_conv = nn.Sequential(*[
                    nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_channels//2, momentum=0.01),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(in_channels, momentum=0.01),
                    ])

            self.post_cat = nn.Sequential(*[make_fc(self.out_channels * 3, self.out_channels), nn.ReLU(inplace=True),
                                            make_fc(self.out_channels, self.out_channels), nn.ReLU(inplace=True)])
            self.head_proj = nn.Sequential(*[make_fc(self.out_channels, self.out_channels), nn.ReLU(inplace=True)])
            self.tail_proj = nn.Sequential(*[make_fc(self.out_channels, self.out_channels), nn.ReLU(inplace=True)])
            self.vis_cls = make_fc(self.out_channels, self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)

    def forward(self, x, windows):
        """
        Arguments:
            x (list[Tensor]): feature-maps from possibly several levels
            windows (list[BoxList]): extracted windows of all images
        """
        device = x[0].device

        assert self.use_pairs or self.use_window_box

        window_scores = None

        if self.use_pairs:
            # 1. subject, object visual features, and union visual, spatial features
            head_proposals, tail_proposals = [], []
            union_proposals = []
            rect_inputs = []
            num_rels_by_image = []
            for window_single_image in windows:
                clipped_head_bboxlist = window_single_image.get_field("clipped_head_bbox").copy()
                clipped_tail_bboxlist = window_single_image.get_field("clipped_tail_bbox").copy()
                clipped_union_bboxlist = boxlist_union(clipped_head_bboxlist, clipped_tail_bboxlist)
                union_proposals.append(clipped_union_bboxlist)
                head_proposals.append(clipped_head_bboxlist)
                tail_proposals.append(clipped_tail_bboxlist)
                num_rels_by_image.append(len(clipped_union_bboxlist))
                if self.spatial_on:
                    num_rel = len(clipped_union_bboxlist)
                    # use range to construct rectangle, sized (rect_size, rect_size)
                    dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
                    dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
                    # resize bbox to the scale rect_size
                    clipped_head_bboxlist = clipped_head_bboxlist.resize((self.rect_size, self.rect_size))
                    clipped_tail_bboxlist = clipped_tail_bboxlist.resize((self.rect_size, self.rect_size))
                    head_rect = ((dummy_x_range >= clipped_head_bboxlist.bbox[:,0].floor().view(-1,1,1).long()) & \
                                (dummy_x_range <= clipped_head_bboxlist.bbox[:,2].ceil().view(-1,1,1).long()) & \
                                (dummy_y_range >= clipped_head_bboxlist.bbox[:,1].floor().view(-1,1,1).long()) & \
                                (dummy_y_range <= clipped_head_bboxlist.bbox[:,3].ceil().view(-1,1,1).long())).float()
                    tail_rect = ((dummy_x_range >= clipped_tail_bboxlist.bbox[:,0].floor().view(-1,1,1).long()) & \
                                (dummy_x_range <= clipped_tail_bboxlist.bbox[:,2].ceil().view(-1,1,1).long()) & \
                                (dummy_y_range >= clipped_tail_bboxlist.bbox[:,1].floor().view(-1,1,1).long()) & \
                                (dummy_y_range <= clipped_tail_bboxlist.bbox[:,3].ceil().view(-1,1,1).long())).float()

                    rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 4, rect_size, rect_size)
                    rect_inputs.append(rect_input)

            # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
            union_vis_features = self.feature_extractor.pooler(x, union_proposals)

            head_vis_features = self.feature_extractor(x, head_proposals)
            tail_vis_features = self.feature_extractor(x, tail_proposals)

            if not self.spatial_on:
                union_features = self.feature_extractor.forward_without_pool(union_vis_features)
            else:
                # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
                rect_inputs = torch.cat(rect_inputs, dim=0)
                rect_features = self.rect_conv(rect_inputs)

                # merge two parts
                if self.separate_spatial:
                    region_features = self.feature_extractor.forward_without_pool(union_vis_features)
                    spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
                    union_features = (region_features, spatial_features)
                else:
                    union_features = union_vis_features + rect_features
                    union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

            # 2. gather
            ht_vis_features = self.post_cat(torch.cat([head_vis_features, tail_vis_features, union_features], 1))
            rel_scores = self.vis_cls(ht_vis_features + self.head_proj(head_vis_features) + self.tail_proj(tail_vis_features))

            # 3. semantic feautres
            if self.use_semantics:
                all_pair_cls = []
                for window_single_image in windows:
                    pair_cls = window_single_image.get_field("relation")[:, :2]
                    all_pair_cls.append(pair_cls)
                freq_scores = self.freq_bias(torch.cat(all_pair_cls, 0))
                rel_scores = rel_scores + freq_scores

            # 4. transform to window scores
            nums = []
            for window_single_image in windows:
                num = window_single_image.get_field("num")
                if isinstance(num, torch.Tensor):
                    num = num.cpu().numpy().tolist()
                nums += num
            rel_scores = rel_scores.split(nums)
            pooled_window_scores = []
            for s in rel_scores:
                pooled_window_scores.append(torch.max(s, 0, keepdim=True)[0])
            pooled_window_scores = torch.cat(pooled_window_scores, 0)

            window_scores = pooled_window_scores

        # 5. extract window features
        if self.use_window_box:
            window_features = self.feature_extractor(x, windows)
            win_scores = self.win_cls(window_features)
            if window_scores is not None:
                window_scores = window_scores + win_scores
            else:
                window_scores = win_scores
        return window_scores


def make_roi_window_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_WINDOW_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_WINDOW_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
