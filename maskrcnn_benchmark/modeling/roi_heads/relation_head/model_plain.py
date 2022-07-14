# ---------------------------------------------------------------
# model_mlp.py
# Set-up time: 2021/11/30 15:40
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

# modified from https://github.com/rowanz/neural-motifs
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot


class PlainContext(nn.Module):
    def __init__(self, config, num_obj, num_rel, in_channels, hidden_dim=512, num_iter=3):
        super(PlainContext, self).__init__()
        self.cfg = config
        self.num_obj = num_obj
        self.num_rel = num_rel
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter
        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.obj_fc = make_fc(hidden_dim, self.num_obj)

        self.obj_unary = make_fc(in_channels, hidden_dim)

        # self.edge_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        # self.node_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        #
        # self.sub_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        # self.obj_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        # self.out_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())
        # self.in_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def forward(self, x, proposals, rel_pair_idxs, logger=None):
        num_objs = [len(b) for b in proposals]

        obj_rep = F.relu(self.obj_unary(x))

        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.num_obj)
        else:
            obj_dists = self.obj_fc(obj_rep)


        return obj_dists, obj_rep
