# ---------------------------------------------------------------
# _utils.py
# Set-up time: 2022/1/6 21:22
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------

import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx_output = r_mat_inv.dot(mx)
    return mx_output


def adj_normalize(adj):
    #adj = adj + adj.T * (adj.T > adj) - adj * (adj.T > adj)
    # adj = adj + adj.T
    adj = normalize(adj + np.eye(adj.shape[0]))
    return adj