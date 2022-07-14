# ---------------------------------------------------------------
# window_head.py
# Set-up time: 2021/12/28 15:30
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
from torch import nn
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_clip_and_change_coordinate_to_window
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist
from .roi_window_relation_feature_extractors import make_roi_window_relation_feature_extractor
from .loss import make_roi_window_relation_loss_evaluator
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data import get_dataset_statistics


class ROIWindowHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIWindowHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in union_feature_extractor
        self.feature_extractor = make_roi_window_relation_feature_extractor(cfg, in_channels)
        self.loss_evaluator = make_roi_window_relation_loss_evaluator(cfg)

        if self.cfg.MODEL.ROI_WINDOW_HEAD.MIX_TYPE == "freq":
            statistics = get_dataset_statistics(cfg)
            freq = statistics['fg_matrix'].float().sum([0, 1])[None]  # 1, 51
            freq = torch.log(freq / freq.sum() + 1e-3)
            self.register_buffer("freq", freq)

    def forward(self, features, proposals, obj_logits, rel_logits, rel_pair_idxs, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            assert targets is not None
            windows = [t.get_field("window") for t in targets]

            window_scores = self.feature_extractor(features, windows)
            loss_window = self.loss_evaluator(window_scores, windows)
            output_losses = dict(loss_window=loss_window)
        else:
            output_losses = {}

        # v2: rectified
        # for both training and testing
        # get the relation refined logits
        scores_multi_scales_all_imgs = [[] for _ in range(len(proposals))]

        mask = torch.ones(len(proposals), device=features[0].device)
        for i, r in enumerate(rel_pair_idxs):
            if len(r) == 0:
                mask[i] = 0
        mask = mask.bool()
        masked_features = tuple([feat[mask] for feat in features])

        for i in range(self.cfg.MODEL.ROI_WINDOW_HEAD.NUM_WINDOW_REFINE):
            win_scale_i_allboxes = []
            for proposal, obj_logit, rel_logit, rel_pair_idx in zip(proposals, obj_logits, rel_logits, rel_pair_idxs):
                w, h = proposal.size
                if len(rel_pair_idx) == 0:
                    continue
                head_proposal = proposal[rel_pair_idx[:, 0]]
                tail_proposal = proposal[rel_pair_idx[:, 1]]
                union_proposal = boxlist_union(head_proposal, tail_proposal)
                union_bbox = union_proposal.bbox
                left_interval = union_bbox[:, 0:1] / self.cfg.MODEL.ROI_WINDOW_HEAD.NUM_WINDOW_REFINE
                right_interval = (w - 1 - union_bbox[:, 2:3]) / self.cfg.MODEL.ROI_WINDOW_HEAD.NUM_WINDOW_REFINE
                up_interval = union_bbox[:, 1:2] / self.cfg.MODEL.ROI_WINDOW_HEAD.NUM_WINDOW_REFINE
                bottom_interval = (h - 1 - union_bbox[:, 3:4]) / self.cfg.MODEL.ROI_WINDOW_HEAD.NUM_WINDOW_REFINE
                win_left = (union_bbox[:, 0:1] - (i + 1) * left_interval).clamp(min=0, max=w-1)
                win_right = (union_bbox[:, 2:3] + (i + 1) * right_interval).clamp(min=0, max=w-1)
                win_up = (union_bbox[:, 1:2] - (i + 1) * up_interval).clamp(min=0, max=h-1)
                win_bottom = (union_bbox[:, 3:4] + (i + 1) * bottom_interval).clamp(min=0, max=h-1)
                win_scale_i = torch.cat([win_left, win_up, win_right, win_bottom], dim=1)
                win_scale_i_boxlist = BoxList(win_scale_i, (w, h), 'xyxy')
                win_scale_i_allboxes.append(win_scale_i_boxlist)
            num = [len(b) for b in win_scale_i_allboxes]
            scores_scale_i = self.feature_extractor(masked_features, win_scale_i_allboxes)
            for j, scores_scale_i_img_j in enumerate(scores_scale_i.split(num)):
                scores_multi_scales_all_imgs[j].append(scores_scale_i_img_j)

        for i in range(len(scores_multi_scales_all_imgs)):
            if len(scores_multi_scales_all_imgs[i]) > 0:
                scores_multi_scales_all_imgs[i] = torch.stack(scores_multi_scales_all_imgs[i], 0).mean(dim=0)
                #scores_multi_scales_all_imgs[i] = torch.cat(scores_multi_scales_all_imgs[i], 1)
        # filter empty:
        filt_scores_multi_scales_all_imgs = []
        for s in scores_multi_scales_all_imgs:
            if len(s) > 0:
                filt_scores_multi_scales_all_imgs.append(s)
        scores_multi_scales_all_imgs = torch.cat(filt_scores_multi_scales_all_imgs, 0)
        if self.training:
            if self.cfg.MODEL.ROI_WINDOW_HEAD.MIX_TYPE == "shuffle":
                scores_multi_scales_all_imgs = scores_multi_scales_all_imgs[torch.randperm(len(scores_multi_scales_all_imgs)).to(scores_multi_scales_all_imgs.device)]
            elif self.cfg.MODEL.ROI_WINDOW_HEAD.MIX_TYPE == "random":
                scores_multi_scales_all_imgs = torch.randn_like(scores_multi_scales_all_imgs)
            elif self.cfg.MODEL.ROI_WINDOW_HEAD.MIX_TYPE == "random_uniform":
                scores_multi_scales_all_imgs = torch.rand_like(scores_multi_scales_all_imgs)
            elif self.cfg.MODEL.ROI_WINDOW_HEAD.MIX_TYPE == "freq":
                scores_multi_scales_all_imgs = self.freq
        return scores_multi_scales_all_imgs, output_losses




def build_roi_window_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIWindowHead(cfg, in_channels)
