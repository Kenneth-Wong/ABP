# ---------------------------------------------------------------
# window_sampling.py
# Set-up time: 2021/12/21 14:48
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import os
import sys
import torch
import h5py
import json
import logging
from PIL import Image
import argparse
import pickle
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm
import random
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_union_area, boxlist_union
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_clip_and_change_coordinate_to_window
from multiprocessing import Pool


def single_process(target, num_classes, min_size, window_num, max_rel_num):
    relation_matrix = target.get_field("relation")
    tgt_pair_idxs = torch.nonzero(relation_matrix > 0)
    tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
    tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
    tgt_rel_labs = relation_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
    # relation_ = torch.from_numpy(relation).long()
    head_target = target[tgt_head_idxs]
    tail_target = target[tgt_tail_idxs]
    union_target = boxlist_union(head_target, tail_target)
    union_target.add_field("labels", tgt_rel_labs)
    union_target.add_field("head_bbox", head_target)
    union_target.add_field("tail_bbox", tail_target)

    # w, h = union_target.size
    # return BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')

    w, h = union_target.size
    if len(union_target) == 0:  # because of the bilevel sampling, there may be no relations
        return BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')

    windows = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    window_soft_labels = torch.zeros(0, num_classes).float()
    window_hard_labels = torch.zeros(0, ).long()
    all_clipped_head_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_clipped_tail_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    reffered_pair_num = []

    head_bbox, tail_bbox = union_target.get_field("head_bbox"), union_target.get_field("tail_bbox")
    labels = union_target.get_field("labels")  # groundtruth labels

    while windows.bbox.size(0) < window_num:
        # 1. generate qualified windows
        win_widths = torch.rand(window_num) * (w - min_size) + min_size  # [min_size, w)
        win_heights = torch.rand(window_num) * (h - min_size) + min_size  # [min_size, h)
        cx = torch.rand(window_num) * w  # [0, w)
        cy = torch.rand(window_num) * h  # [0, h)
        x1 = (cx - win_widths / 2).clamp(min=0.0, max=w - 1)
        x2 = (cx + win_widths / 2 - 1).clamp(min=0.0, max=w - 1)
        y1 = (cy - win_heights / 2).clamp(min=0.0, max=h - 1)
        y2 = (cy + win_heights / 2 - 1).clamp(min=0.0, max=h - 1)
        x2[(x2 - x1 + 1 < min_size) & (x1 == 0)] = min_size - 1
        x1[(x2 - x1 + 1 < min_size) & (x2 == w - 1)] = w - min_size
        y2[(y2 - y1 + 1 < min_size) & (y1 == 0)] = min_size - 1
        y1[(y2 - y1 + 1 < min_size) & (y2 == h - 1)] = h - min_size
        assert (x1 >= 0).all() and (x1 < w).all() and (x2 >= 0).all() and (x2 < w).all() and \
               (y1 >= 0).all() and (y1 < h).all() and (y2 >= 0).all() and (y2 < h).all() and \
               (x2 - x1 + 1 >= min_size).all() and (y2 - y1 + 1 >= min_size).all()

        tmp_windows = BoxList(torch.cat((x1[:, None], y1[:, None], x2[:, None], y2[:, None]), dim=1), (w, h), 'xyxy')

        # 2.check and construct soft label for each window
        keep_window_idx = []
        for i in range(len(tmp_windows)):
            window = tmp_windows[i:i + 1].bbox
            # get the prob for bg: using all the boxes
            # using all the pairs to compute:
            clipped = boxlist_clip_and_change_coordinate_to_window(head_bbox, tail_bbox, window, labels)
            if clipped is None:
                continue

            # all the overlapping relations with the current window
            clipped_head_bboxes, clipped_tail_bboxes, norm_clipped_head_bboxes, norm_clipped_tail_bboxes, indices = \
                clipped

            # only randomly keep #max_rel_num rels
            if len(indices) > max_rel_num:
                selected = np.sort(np.random.choice(list(range(len(indices))), max_rel_num))
                indices = indices[selected]
                clipped_head_bboxes = clipped_head_bboxes[selected]
                clipped_tail_bboxes = clipped_tail_bboxes[selected]
                norm_clipped_head_bboxes = norm_clipped_head_bboxes[selected]
                norm_clipped_tail_bboxes = norm_clipped_tail_bboxes[selected]

            hard_labels = labels[indices]
            clipped_all = cat_boxlist([norm_clipped_head_bboxes, norm_clipped_tail_bboxes])
            area_total = boxlist_union_area(clipped_all)

            label_bg = np.sqrt(max(0., 1. - area_total))
            if label_bg == 1.:
                continue

            soft_label = torch.zeros(num_classes)
            soft_label[0] = label_bg
            # using boxes of each predicate
            for predicate in torch.unique(hard_labels):
                # clipped = boxlist_clip_and_change_coordinate_to_window(cls_head_bbox, cls_tail_bbox, window, labels=None)
                # if clipped is None:
                #     continue
                cls_indices = torch.nonzero(hard_labels == predicate).view(-1)
                cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes = norm_clipped_head_bboxes[cls_indices], \
                                                                             norm_clipped_tail_bboxes[cls_indices]
                clipped_all = cat_boxlist([cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes])
                area_total = boxlist_union_area(clipped_all)
                soft_label[predicate] = np.sqrt(area_total)

            soft_label = soft_label / torch.sum(soft_label)

            keep_window_idx.append(i)
            window_soft_labels = torch.cat((window_soft_labels, soft_label.view(1, -1)), 0)
            window_hard_labels = torch.cat((window_hard_labels, hard_labels), 0)
            all_clipped_head_bboxes = cat_boxlist([all_clipped_head_bboxes, clipped_head_bboxes])
            all_clipped_tail_bboxes = cat_boxlist([all_clipped_tail_bboxes, clipped_tail_bboxes])
            reffered_pair_num.append(len(indices))

        tmp_windows = tmp_windows[keep_window_idx]
        windows = cat_boxlist([windows, tmp_windows])
        del tmp_windows
        del x1, x2, y1, y2, cx, cy, win_heights, win_widths


    # num_all = len(windows)
    # windows = windows[:window_num]
    # delete_num = sum(reffered_pair_num[-(num_all - window_num):])
    # reffered_pair_num = reffered_pair_num[:window_num]
    # window_soft_labels = window_soft_labels[:window_num]
    # window_hard_labels = window_hard_labels[:-delete_num]
    # all_clipped_head_bboxes = all_clipped_head_bboxes[:-delete_num]
    # all_clipped_tail_bboxes = all_clipped_tail_bboxes[:-delete_num]

    windows.add_field("hard_labels", window_hard_labels)
    windows.add_field("soft_labels", window_soft_labels)
    windows.add_field("clipped_head_bbox", all_clipped_head_bboxes)
    windows.add_field("clipped_tail_bbox", all_clipped_tail_bboxes)
    windows.add_field("num", torch.tensor(reffered_pair_num).long())
    return windows


def multiproc_sample_window(dataset, num_classes, logger):
    logger.info("Using sampled windows. ")
    sampled_windows_path = os.path.join(cfg.OUTPUT_DIR, "sampled_windows.pkl")
    if os.path.exists(sampled_windows_path):
        logger.info("load windows from " + sampled_windows_path)
        with open(sampled_windows_path, "rb") as f:
            # torch.save(self.windows, os.path.join(cfg.OUTPUT_DIR, "sampled_windows.pt"))
            all_windows = pickle.load(f)
        # all_windows = torch.load(sampled_windows_path, map_location=torch.device("cpu"))
        return all_windows
    else:
        logger.info("generate the windows.")
        min_size = cfg.MODEL.MIN_WINDOW_SIZE
        window_num = cfg.MODEL.WINDOW_NUM
        max_rel_num = cfg.MODEL.MAX_REL_NUM
        pool = Pool(processes=32)
        all_windows = []
        for i in tqdm(range(len(dataset))):
            target = dataset.get_groundtruth(i, inner_idx=False)
            all_windows.append(pool.apply_async(single_process, (target, num_classes, min_size, window_num, max_rel_num)))
        pool.close()
        pool.join()
        all_windows = [r.get() for r in all_windows]
        return all_windows



def sample_window(dataset, num_classes, logger):
    logger.info("Using sampled windows. ")
    sampled_windows_path = os.path.join(cfg.OUTPUT_DIR, "sampled_windows.pkl")
    if os.path.exists(sampled_windows_path):
        logger.info("load windows from " + sampled_windows_path)
        with open(sampled_windows_path, "rb") as f:
            # torch.save(self.windows, os.path.join(cfg.OUTPUT_DIR, "sampled_windows.pt"))
            all_windows = pickle.load(f)
        #all_windows = torch.load(sampled_windows_path, map_location=torch.device("cpu"))
        return all_windows
    else:
        logger.info("generate the windows.")
        min_size = cfg.MODEL.MIN_WINDOW_SIZE
        window_num = cfg.MODEL.WINDOW_NUM
        max_rel_num = cfg.MODEL.MAX_REL_NUM

        all_windows = []
        for i in tqdm(range(len(dataset))):
            # here, the idx_list maybe resampled, so the inner_idx is False
            target = dataset.get_groundtruth(i, inner_idx=False)
            relation_matrix = target.get_field("relation")
            tgt_pair_idxs = torch.nonzero(relation_matrix > 0)
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = relation_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
            # relation_ = torch.from_numpy(relation).long()
            head_target = target[tgt_head_idxs]
            tail_target = target[tgt_tail_idxs]
            union_target = boxlist_union(head_target, tail_target)
            union_target.add_field("labels", tgt_rel_labs)
            union_target.add_field("head_bbox", head_target)
            union_target.add_field("tail_bbox", tail_target)

            w, h = union_target.size
            if len(union_target) == 0: # because of the bilevel sampling, there may be no relations
                all_windows.append(BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy'))
                continue

            windows = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
            window_soft_labels  = torch.zeros(0, num_classes).float()
            window_hard_labels = torch.zeros(0,).long()
            all_clipped_head_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
            all_clipped_tail_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
            reffered_pair_num = []

            head_bbox, tail_bbox = union_target.get_field("head_bbox"), union_target.get_field("tail_bbox")
            labels = union_target.get_field("labels")  # groundtruth labels

            while windows.bbox.size(0) < window_num:
                # 1. generate qualified windows
                win_widths = torch.rand(window_num) * (w - min_size) + min_size  # [min_size, w)
                win_heights = torch.rand(window_num) * (h - min_size) + min_size  # [min_size, h)
                cx = torch.rand(window_num) * w   # [0, w)
                cy = torch.rand(window_num) * h   # [0, h)
                x1 = (cx - win_widths / 2).clamp(min=0.0, max=w-1)
                x2 = (cx + win_widths / 2 - 1).clamp(min=0.0, max=w-1)
                y1 = (cy - win_heights / 2).clamp(min=0.0, max=h-1)
                y2 = (cy + win_heights / 2 - 1).clamp(min=0.0, max=h-1)
                x2[(x2 - x1 + 1 < min_size) & (x1 == 0)] = min_size - 1
                x1[(x2 - x1 + 1 < min_size) & (x2 == w-1)] = w - min_size
                y2[(y2 - y1 + 1 < min_size) & (y1 == 0)] = min_size - 1
                y1[(y2 - y1 + 1 < min_size) & (y2 == h-1)] = h - min_size
                assert (x1 >= 0).all() and (x1 < w).all() and (x2 >= 0).all() and (x2 < w).all() and \
                       (y1 >= 0).all() and (y1 < h).all() and (y2 >= 0).all() and (y2 < h).all() and \
                       (x2 - x1 + 1 >= min_size).all() and (y2 - y1 + 1 >= min_size).all()

                tmp_windows = BoxList(torch.cat((x1[:, None], y1[:, None], x2[:, None], y2[:, None]), dim=1), (w, h), 'xyxy')

                # 2.check and construct soft label for each window
                keep_window_idx = []
                for i in range(len(tmp_windows)):
                    window = tmp_windows[i:i+1].bbox
                    # get the prob for bg: using all the boxes
                    # using all the pairs to compute:
                    clipped = boxlist_clip_and_change_coordinate_to_window(head_bbox, tail_bbox, window, labels)
                    if clipped is None:
                        continue

                    # all the overlapping relations with the current window
                    clipped_head_bboxes, clipped_tail_bboxes, norm_clipped_head_bboxes, norm_clipped_tail_bboxes, indices = \
                        clipped

                    # only randomly keep #max_rel_num rels
                    if len(indices) > max_rel_num:
                        selected = np.sort(np.random.choice(list(range(len(indices))), max_rel_num))
                        indices = indices[selected]
                        clipped_head_bboxes = clipped_head_bboxes[selected]
                        clipped_tail_bboxes = clipped_tail_bboxes[selected]
                        norm_clipped_head_bboxes = norm_clipped_head_bboxes[selected]
                        norm_clipped_tail_bboxes = norm_clipped_tail_bboxes[selected]


                    hard_labels = labels[indices]
                    clipped_all = cat_boxlist([norm_clipped_head_bboxes, norm_clipped_tail_bboxes])
                    area_total = boxlist_union_area(clipped_all)

                    label_bg = np.sqrt(max(0., 1. - area_total))
                    if label_bg == 1.:
                        continue

                    soft_label = torch.zeros(num_classes)
                    soft_label[0] = label_bg
                    # using boxes of each predicate
                    for predicate in torch.unique(hard_labels):
                        # clipped = boxlist_clip_and_change_coordinate_to_window(cls_head_bbox, cls_tail_bbox, window, labels=None)
                        # if clipped is None:
                        #     continue
                        cls_indices = torch.nonzero(hard_labels==predicate).view(-1)
                        cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes = norm_clipped_head_bboxes[cls_indices], \
                                                                                     norm_clipped_tail_bboxes[cls_indices]
                        clipped_all = cat_boxlist([cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes])
                        area_total = boxlist_union_area(clipped_all)
                        soft_label[predicate] = np.sqrt(area_total)

                    soft_label = soft_label / torch.sum(soft_label)

                    keep_window_idx.append(i)
                    window_soft_labels = torch.cat((window_soft_labels, soft_label.view(1, -1)), 0)
                    window_hard_labels = torch.cat((window_hard_labels, hard_labels), 0)
                    all_clipped_head_bboxes = cat_boxlist([all_clipped_head_bboxes, clipped_head_bboxes])
                    all_clipped_tail_bboxes = cat_boxlist([all_clipped_tail_bboxes, clipped_tail_bboxes])
                    reffered_pair_num.append(len(indices))

                tmp_windows = tmp_windows[keep_window_idx]
                windows = cat_boxlist([windows, tmp_windows])

            num_all = len(windows)
            windows = windows[:window_num]
            delete_num = sum(reffered_pair_num[-(num_all-window_num):])
            reffered_pair_num = reffered_pair_num[:window_num]
            window_soft_labels = window_soft_labels[:window_num]
            window_hard_labels = window_hard_labels[:-delete_num]
            all_clipped_head_bboxes = all_clipped_head_bboxes[:-delete_num]
            all_clipped_tail_bboxes = all_clipped_tail_bboxes[:-delete_num]

            windows.add_field("hard_labels", window_hard_labels)
            windows.add_field("soft_labels", window_soft_labels)
            windows.add_field("clipped_head_bbox", all_clipped_head_bboxes)
            windows.add_field("clipped_tail_bbox", all_clipped_tail_bboxes)
            windows.add_field("num", torch.tensor(reffered_pair_num).long())
            all_windows.append(windows)
        return all_windows
