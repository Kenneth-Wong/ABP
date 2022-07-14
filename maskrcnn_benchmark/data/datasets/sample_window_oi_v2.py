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
from maskrcnn_benchmark.data.datasets.open_image import load_annotations
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.data import datasets as D

np.random.seed(666)
torch.manual_seed(666)


def single_process(head_bbox, tail_bbox, obj_bbox, labels, num_classes, min_size, window_num):

    w, h = head_bbox.size

    windows = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_head_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_tail_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_clipped_head_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_clipped_tail_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_norm_clipped_head_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_norm_clipped_tail_bboxes = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    all_indices = torch.zeros(0, ).long()
    reffered_pair_num = []
    
    num_tries = 0
    while windows.bbox.size(0) < window_num:
        if num_tries > 5 and len(windows) == 0:
            return windows
        num_tries += 1

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
        # assert (x1 >= 0).all() and (x1 < w).all() and (x2 >= 0).all() and (x2 < w).all() and \
        #        (y1 >= 0).all() and (y1 < h).all() and (y2 >= 0).all() and (y2 < h).all() and \
        #        (x2 - x1 + 1 >= min_size).all() and (y2 - y1 + 1 >= min_size).all()

        tmp_windows = BoxList(torch.cat((x1[:, None], y1[:, None], x2[:, None], y2[:, None]), dim=1), (w, h), 'xyxy')

        # 2.check and construct soft label for each window
        keep_window_idx = []
        for i in range(len(tmp_windows)):
            window = tmp_windows[i:i + 1].bbox
            wx1, wy1, wx2, wy2 = window[0]
            win_height, win_width = float(wy2) - float(wy1) + 1, float(wx2) - float(wx1) + 1
            if not(float(wx1)>=0 and float(wx1)<=w-1 and float(wx2)>=0 and float(wx2)<=w-1 and
            float(wy1)>=0 and float(wy1)<=h-1 and float(wy2)>=0 and float(wy2)<=h-1 and
            win_height >= min_size and win_width >= min_size):
                print(window.bbox, w, h)
                continue
            # get the prob for bg: using all the boxes
            # using all the pairs to compute:
            clipped = boxlist_clip_and_change_coordinate_to_window(head_bbox, tail_bbox, window)
            if clipped is None:
                continue

            # all the overlapping relations with the current window
            clipped_head_bboxes, clipped_tail_bboxes, norm_clipped_head_bboxes, norm_clipped_tail_bboxes, indices = \
                clipped

            head_bboxes = head_bbox[indices]
            tail_bboxes = tail_bbox[indices]

            keep_window_idx.append(i)
            all_head_bboxes = cat_boxlist([all_head_bboxes, head_bboxes])
            all_tail_bboxes = cat_boxlist([all_tail_bboxes, tail_bboxes])
            all_clipped_head_bboxes = cat_boxlist([all_clipped_head_bboxes, clipped_head_bboxes])
            all_clipped_tail_bboxes = cat_boxlist([all_clipped_tail_bboxes, clipped_tail_bboxes])
            all_norm_clipped_head_bboxes = cat_boxlist([all_norm_clipped_head_bboxes, norm_clipped_head_bboxes])
            all_norm_clipped_tail_bboxes = cat_boxlist([all_norm_clipped_tail_bboxes, norm_clipped_tail_bboxes])
            all_indices = torch.cat((all_indices, indices), 0)
            reffered_pair_num.append(len(indices))

        tmp_windows = tmp_windows[keep_window_idx]
        windows = cat_boxlist([windows, tmp_windows])
        del tmp_windows
        del x1, x2, y1, y2, cx, cy, win_heights, win_widths


    num_all = len(windows)
    if num_all > window_num: # IMPORTANT!!!!: if num_all == window_num, the following codes not work!!!
        windows = windows[:window_num]
        delete_num = sum(reffered_pair_num[-(num_all - window_num):])
        reffered_pair_num = reffered_pair_num[:window_num]
        all_head_bboxes = all_head_bboxes[:-delete_num]
        all_tail_bboxes = all_tail_bboxes[:-delete_num]
        all_clipped_head_bboxes = all_clipped_head_bboxes[:-delete_num]
        all_clipped_tail_bboxes = all_clipped_tail_bboxes[:-delete_num]
        all_norm_clipped_head_bboxes = all_norm_clipped_head_bboxes[:-delete_num]
        all_norm_clipped_tail_bboxes = all_norm_clipped_tail_bboxes[:-delete_num]
        all_indices = all_indices[:-delete_num]

    assert sum(reffered_pair_num) == len(all_head_bboxes)

    windows.add_field("head_bbox", all_head_bboxes)
    windows.add_field("tail_bbox", all_tail_bboxes)
    windows.add_field("clipped_head_bbox", all_clipped_head_bboxes)
    windows.add_field("clipped_tail_bbox", all_clipped_tail_bboxes)
    windows.add_field("norm_clipped_head_bbox", all_norm_clipped_head_bboxes)
    windows.add_field("norm_clipped_tail_bbox", all_norm_clipped_tail_bboxes)
    windows.add_field("indices", all_indices)
    windows.add_field("num", torch.tensor(reffered_pair_num).long())
    return windows


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ann_file', default='datasets/openimages/open_image_v%d/annotations/vrd-train-anno.json')
    #parser.add_argument('--img_dir', default='datasets/openimages/open_image_v%d/images')
    parser.add_argument('--num_classes', default=602)
    parser.add_argument('--version', type=int, default=6)
    parser.add_argument('--min_size', default=32)
    parser.add_argument('--window_num', default=32, type=int)
    parser.add_argument('--h5_output',
                        default='debug.h5',
                        help='Path to output HDF5 file')
    args = parser.parse_args()

    root = "datasets/openimages/open_image_v%d" % args.version
    ann_file = os.path.join(root, "annotations/vrd-train-anno.json")
    img_dir = os.path.join(root, "images")
    args.h5_output = os.path.join(root, args.h5_output)

    boxes, gt_classes, relationships, img_info = load_annotations(ann_file, img_dir, -1, "train", True)

    # check
    # for i, (box, info_i) in tqdm(enumerate(zip(boxes, img_info))):
    #     w, h = info_i['width'], info_i['height']
    #     if not (np.all(box[:, :2] >= 0) and np.all(box[:, 2] < w) and np.all(box[:, 3] < h)):
    #         print(i)

    f = h5py.File(args.h5_output, 'w')

    num_images = len(boxes)

    # begin the main loop:
    all_relation_labels = []
    im_to_first_all_relation = np.ones(num_images, dtype=np.int32) * -1
    im_to_last_all_relation = np.ones(num_images, dtype=np.int32) * -1
    rel_counter = 0

    all_windows = []
    all_referred_num = []
    im_to_first_all_window = np.ones(num_images, dtype=np.int32) * -1
    im_to_last_all_window = np.ones(num_images, dtype=np.int32) * -1
    window_counter = 0

    all_head_bbox = []
    all_tail_bbox = []
    all_clipped_head_bbox = []
    all_clipped_tail_bbox = []
    all_norm_clipped_head_bbox = []
    all_norm_clipped_tail_bbox = []
    all_indices = []
    im_to_first_referred = np.ones(num_images, dtype=np.int32) * -1
    im_to_last_refferred = np.ones(num_images, dtype=np.int32) * -1
    rel_box_counter = 0

    for i in tqdm(range(num_images)):
        w, h = img_info[i]['width'], img_info[i]['height']
        boxes_i = boxes[i]

        gt_classes_i = gt_classes[i]

        rels = relationships[i]  # (num_rel, 3), representing sub, obj, and pred

        # filter duplicate rels
        all_rel_sets = defaultdict(list)
        for (o0, o1, r) in rels:
            all_rel_sets[(o0, o1)].append(r)
        rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
        rels = np.array(rels, dtype=np.int32)


        size = (w, h)
        head_bbox = BoxList(boxes_i[rels[:, 0], :], size, 'xyxy')
        tail_bbox = BoxList(boxes_i[rels[:, 1], :], size, 'xyxy')
        obj_bbox = BoxList(boxes_i, size, 'xyxy')

        windows_i = single_process(head_bbox, tail_bbox, obj_bbox, torch.from_numpy(gt_classes_i).long(),
                                   args.num_classes, args.min_size, args.window_num)
        
        if len(windows_i) == 0: # failed sampling
            print("failed sampling on %d. skip." % i)
            continue
        
        im_to_first_all_relation[i] = rel_counter
        all_relation_labels.append(np.column_stack((gt_classes_i[rels[:, 0]][:, None],gt_classes_i[rels[:, 1]][:, None],
                                                    rels[:, -1][:, None])))
        rel_counter += len(rels)
        im_to_last_all_relation[i] = rel_counter - 1

        im_to_first_all_window[i] = window_counter
        all_windows.append(windows_i.bbox.numpy())
        all_referred_num.append(windows_i.get_field("num").numpy())
        window_counter += len(windows_i)
        im_to_last_all_window[i] = window_counter - 1

        im_to_first_referred[i] = rel_box_counter
        all_head_bbox.append(windows_i.get_field("head_bbox").bbox.numpy())
        all_tail_bbox.append(windows_i.get_field("tail_bbox").bbox.numpy())
        all_clipped_head_bbox.append(windows_i.get_field("clipped_head_bbox").bbox.numpy())
        all_clipped_tail_bbox.append(windows_i.get_field("clipped_tail_bbox").bbox.numpy())
        all_norm_clipped_head_bbox.append(windows_i.get_field("norm_clipped_head_bbox").bbox.numpy())
        all_norm_clipped_tail_bbox.append(windows_i.get_field("norm_clipped_tail_bbox").bbox.numpy())
        all_indices.append(windows_i.get_field("indices").numpy())
        rel_box_counter += len(windows_i.get_field("head_bbox"))
        im_to_last_refferred[i] = rel_box_counter - 1

    f.create_dataset("window", data=np.vstack(all_windows).astype(np.float32))
    f.create_dataset("referred_num", data=np.hstack(all_referred_num).astype(np.int32))
    f.create_dataset("im_to_first_window", data=im_to_first_all_window)
    f.create_dataset("im_to_last_window", data=im_to_last_all_window)

    f.create_dataset("relation_label", data=np.vstack(all_relation_labels).astype(np.int32))
    f.create_dataset("im_to_first_relation", data=im_to_first_all_relation)
    f.create_dataset("im_to_last_relation", data=im_to_last_all_relation)

    f.create_dataset("head_bbox", data=np.vstack(all_head_bbox).astype(np.float32))
    f.create_dataset("tail_bbox", data=np.vstack(all_tail_bbox).astype(np.float32))
    f.create_dataset("clipped_head_bbox", data=np.vstack(all_clipped_head_bbox).astype(np.float32))
    f.create_dataset("clipped_tail_bbox", data=np.vstack(all_clipped_tail_bbox).astype(np.float32))
    f.create_dataset("norm_clipped_head_bbox", data=np.vstack(all_norm_clipped_head_bbox).astype(np.float32))
    f.create_dataset("norm_clipped_tail_bbox", data=np.vstack(all_norm_clipped_tail_bbox).astype(np.float32))
    f.create_dataset("referred_indice", data=np.hstack(all_indices).astype(np.int32))
    f.create_dataset("im_to_first_referred", data=im_to_first_referred)
    f.create_dataset("im_to_last_referred", data=im_to_last_refferred)
    f.close()

main()
