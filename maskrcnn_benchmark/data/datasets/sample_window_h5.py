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
from maskrcnn_benchmark.data.datasets.visual_genome import VGDataset
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.data import datasets as D

np.random.seed(666)
torch.manual_seed(666)


def load_image_filenames(image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue
        img_info.append(img)
    assert len(img_info) == 108073
    return img_info


def single_process(head_bbox, tail_bbox, labels, num_classes, min_size, window_num, max_rel_num):

    w, h = head_bbox.size

    windows = BoxList(torch.zeros(0, 4).float(), (w, h), 'xyxy')
    #window_soft_labels = torch.zeros(0, num_classes).float()
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

            # only randomly keep #max_rel_num rels
            # if len(indices) > max_rel_num:
            #     selected = np.sort(np.random.choice(list(range(len(indices))), max_rel_num, replace=False))
            #     indices = indices[selected]
            #     head_bbox = head_bbox[selected]
            #     tail_bbox = tail_bbox[selected]
            #     clipped_head_bboxes = clipped_head_bboxes[selected]
            #     clipped_tail_bboxes = clipped_tail_bboxes[selected]
            #     norm_clipped_head_bboxes = norm_clipped_head_bboxes[selected]
            #     norm_clipped_tail_bboxes = norm_clipped_tail_bboxes[selected]
            #
            # hard_labels = labels[indices]
            #
            # # compute soft label
            # clipped_all = cat_boxlist([norm_clipped_head_bboxes, norm_clipped_tail_bboxes])
            # area_total = boxlist_union_area(clipped_all)
            #
            # label_bg = np.sqrt(max(0., 1. - area_total))
            # if label_bg == 1.:
            #     continue
            #
            # soft_label = torch.zeros(num_classes)
            # soft_label[0] = label_bg
            # # using boxes of each predicate
            # for predicate in torch.unique(hard_labels):
            #     cls_indices = torch.nonzero(hard_labels == predicate).view(-1)
            #     cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes = norm_clipped_head_bboxes[cls_indices], \
            #                                                                  norm_clipped_tail_bboxes[cls_indices]
            #     clipped_all = cat_boxlist([cls_norm_clipped_head_bboxes, cls_norm_clipped_tail_bboxes])
            #     area_total = boxlist_union_area(clipped_all)
            #     soft_label[predicate] = np.sqrt(area_total)
            #
            # soft_label = soft_label / torch.sum(soft_label)

            keep_window_idx.append(i)
            #window_soft_labels = torch.cat((window_soft_labels, soft_label.view(1, -1)), 0)
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
        #window_soft_labels = window_soft_labels[:window_num]
        all_head_bboxes = all_head_bboxes[:-delete_num]
        all_tail_bboxes = all_tail_bboxes[:-delete_num]
        all_clipped_head_bboxes = all_clipped_head_bboxes[:-delete_num]
        all_clipped_tail_bboxes = all_clipped_tail_bboxes[:-delete_num]
        all_norm_clipped_head_bboxes = all_norm_clipped_head_bboxes[:-delete_num]
        all_norm_clipped_tail_bboxes = all_norm_clipped_tail_bboxes[:-delete_num]
        all_indices = all_indices[:-delete_num]

    assert sum(reffered_pair_num) == len(all_head_bboxes)

    #windows.add_field("soft_labels", window_soft_labels)
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

    parser.add_argument('--roidb_file', default='datasets/vg/VG-SGG-with-attri.h5')
    parser.add_argument('--image_file', default='datasets/vg/image_data.json')
    parser.add_argument('--dataset_name', default="VG_stanford_filtered_with_attribute_train")
    parser.add_argument('--num_classes', default=51)
    parser.add_argument('--min_size', default=32)
    parser.add_argument('--window_num', default=32, type=int)
    parser.add_argument('--max_rel_num', default=5, type=int)
    parser.add_argument('--h5_output',
                        default='datasets/vg/debug.h5',
                        help='Path to output HDF5 file')

    # OPTIONS
    parser.add_argument('--num_workers', default=5, type=int)
    args = parser.parse_args()

    BOX_SCALE = 1024



    # use all images, ignore the split
    roi_h5 = h5py.File(args.roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_mask = data_split <= 2  # 108073/108073
    split_mask &= roi_h5['img_to_first_box'][:] >= 0  # 105414/108073
    split_mask &= roi_h5['img_to_first_rel'][:] >= 0  # 89169/108073

    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box
    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]

    im_to_first_box = roi_h5['img_to_first_box'][:]
    im_to_last_box = roi_h5['img_to_last_box'][:]
    im_to_first_rel = roi_h5['img_to_first_rel'][:]
    im_to_last_rel = roi_h5['img_to_last_rel'][:]

    img_info = load_image_filenames(args.image_file)

    f = h5py.File(args.h5_output, 'w')

    # begin the main loop:
    all_relation_labels = np.zeros((0, 3), dtype=np.int32)
    im_to_first_all_relation = np.ones(len(split_mask), dtype=np.int32) * -1
    im_to_last_all_relation = np.ones(len(split_mask), dtype=np.int32) * -1

    all_windows = np.zeros((0, 4), dtype=np.float32)
    #all_soft_labels = np.zeros((0, args.num_classes), dtype=np.float32)
    all_referred_num = np.zeros((0,), dtype=np.int32)
    im_to_first_all_window = np.ones(len(split_mask), dtype=np.int32) * -1
    im_to_last_all_window = np.ones(len(split_mask), dtype=np.int32) * -1

    all_head_bbox = np.zeros((0, 4), dtype=np.float32)
    all_tail_bbox = np.zeros((0, 4), dtype=np.float32)
    all_clipped_head_bbox = np.zeros((0, 4), dtype=np.float32)
    all_clipped_tail_bbox = np.zeros((0, 4), dtype=np.float32)
    all_norm_clipped_head_bbox = np.zeros((0, 4), dtype=np.float32)
    all_norm_clipped_tail_bbox = np.zeros((0, 4), dtype=np.float32)
    all_indices = np.zeros((0, ), dtype=np.int32)
    im_to_first_referred = np.ones(len(split_mask), dtype=np.int32) * -1
    im_to_last_refferred = np.ones(len(split_mask), dtype=np.int32) * -1


    for i in tqdm(range(len(split_mask))):
        #if i != 18164:
        #    continue
        if not split_mask[i]:
            continue

        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        w, h = img_info[i]['width'], img_info[i]['height']
        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        boxes_i = boxes_i / BOX_SCALE * max(w, h)
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]


        assert i_rel_start >= 0
        predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
        obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
        assert np.all(obj_idx >= 0)
        assert np.all(obj_idx < boxes_i.shape[0])
        rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred

        # filter duplicate rels
        all_rel_sets = defaultdict(list)
        for (o0, o1, r) in rels:
            all_rel_sets[(o0, o1)].append(r)
        rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
        rels = np.array(rels, dtype=np.int32)


        size = (w, h)
        head_bbox = BoxList(boxes_i[rels[:, 0], :], size, 'xyxy')
        tail_bbox = BoxList(boxes_i[rels[:, 1], :], size, 'xyxy')
        labels = torch.from_numpy(rels[:, -1]).long()

        windows_i = single_process(head_bbox, tail_bbox, labels, args.num_classes, args.min_size,
                                    args.window_num, args.max_rel_num)
        
        if len(windows_i) == 0: # failed sampling
            print("failed sampling on %d. skip." % i)
            continue
        
        im_to_first_all_relation[i] = len(all_relation_labels)
        all_relation_labels = np.vstack((all_relation_labels,
                                         np.column_stack((gt_classes_i[rels[:, 0]][:, None],
                                                          gt_classes_i[rels[:, 1]][:, None],
                                                          rels[:, -1][:, None]))))
        im_to_last_all_relation[i] = len(all_relation_labels) - 1
        
        im_to_first_all_window[i] = len(all_windows)
        all_windows = np.vstack((all_windows, windows_i.bbox.numpy()))
        #all_soft_labels = np.vstack((all_soft_labels, windows_i.get_field("soft_labels").numpy()))
        all_referred_num = np.hstack((all_referred_num, windows_i.get_field("num").numpy()))
        im_to_last_all_window[i] = len(all_windows) - 1

        im_to_first_referred[i] = len(all_clipped_head_bbox)
        all_head_bbox = np.vstack((all_head_bbox, windows_i.get_field("head_bbox").bbox.numpy()))
        all_tail_bbox = np.vstack((all_tail_bbox, windows_i.get_field("tail_bbox").bbox.numpy()))
        all_clipped_head_bbox = np.vstack((all_clipped_head_bbox, windows_i.get_field("clipped_head_bbox").bbox.numpy()))
        all_clipped_tail_bbox = np.vstack((all_clipped_tail_bbox, windows_i.get_field("clipped_tail_bbox").bbox.numpy()))
        all_norm_clipped_head_bbox = np.vstack(
            (all_norm_clipped_head_bbox, windows_i.get_field("norm_clipped_head_bbox").bbox.numpy()))
        all_norm_clipped_tail_bbox = np.vstack(
            (all_norm_clipped_tail_bbox, windows_i.get_field("norm_clipped_tail_bbox").bbox.numpy()))
        all_indices = np.hstack((all_indices, windows_i.get_field("indices").numpy()))
        im_to_last_refferred[i] = len(all_clipped_head_bbox) - 1

    f.create_dataset("window", data=all_windows)
    #f.create_dataset("soft_label", data=all_soft_labels)
    f.create_dataset("referred_num", data=all_referred_num)
    f.create_dataset("im_to_first_window", data=im_to_first_all_window)
    f.create_dataset("im_to_last_window", data=im_to_last_all_window)

    f.create_dataset("relation_label", data=all_relation_labels)
    f.create_dataset("im_to_first_relation", data=im_to_first_all_relation)
    f.create_dataset("im_to_last_relation", data=im_to_last_all_relation)

    f.create_dataset("head_bbox", data=all_head_bbox)
    f.create_dataset("tail_bbox", data=all_tail_bbox)
    f.create_dataset("clipped_head_bbox", data=all_clipped_head_bbox)
    f.create_dataset("clipped_tail_bbox", data=all_clipped_tail_bbox)
    f.create_dataset("norm_clipped_head_bbox", data=all_norm_clipped_head_bbox)
    f.create_dataset("norm_clipped_tail_bbox", data=all_norm_clipped_tail_bbox)
    f.create_dataset("referred_indice", data=all_indices)
    f.create_dataset("im_to_first_referred", data=im_to_first_referred)
    f.create_dataset("im_to_last_referred", data=im_to_last_refferred)
    f.close()

main()
