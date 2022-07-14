# ---------------------------------------------------------------
# loss.py
# Set-up time: 2021/12/29 12:07
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat


class Soft_Label_Regression(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                             .format(x.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        x = self.log_softmax(x)
        loss = torch.sum(- x * target, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


class WindowLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(self, loss_weight):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.loss_weight = loss_weight
        self.criterion = Soft_Label_Regression()

    def __call__(self, window_logits, windows):
        """
        Computes the loss for windows.
        This requires that the subsample method has been called beforehand.
        """
        labels = torch.cat([w.get_field("window_label") for w in windows], 0)
        loss_window = self.criterion(window_logits, labels) * self.loss_weight
        return loss_window



def make_roi_window_relation_loss_evaluator(cfg):

    loss_evaluator = WindowLossComputation(
        cfg.MODEL.ROI_WINDOW_HEAD.LOSS_WEIGHT
    )

    return loss_evaluator