# ---------------------------------------------------------------
# __init__.py
# Set-up time: 2022/2/28 17:22
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
from .vrd_eval import do_vrd_evaluation
from .vg_eval import do_vg_evaluation
from .vgvrd_eval import do_vgvrd_evaluation
import torch
import copy
import os
from maskrcnn_benchmark.utils.miscellaneous import mkdir

def vgvrd_evaluation(
        cfg,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
        **_
):
    # vgvrd_folder = None
    # if output_folder:
    #     vgvrd_folder = os.path.join(output_folder, 'vgvrd')
    #     mkdir(vgvrd_folder)
    # do_vgvrd_evaluation(
    #     cfg=cfg,
    #     dataset=dataset,
    #     predictions=predictions,
    #     output_folder=vgvrd_folder,
    #     logger=logger,
    #     iou_types=iou_types,
    # )

    num_vg_predicates = 51
    # split predictions into two parts
    num_vg = 0
    for fn in dataset.filenames:
        if 'datasets/vg/VG' in fn:
            num_vg += 1
    # vg part
    vg_predictions = predictions[:num_vg]
    vg_dataset = copy.deepcopy(dataset)
    vg_dataset.filenames = vg_dataset.filenames[:num_vg]
    vg_dataset.img_info = vg_dataset.img_info[:num_vg]
    vg_dataset.ind_to_predicates = vg_dataset.ind_to_predicates[:num_vg_predicates]
    vg_dataset.gt_boxes = vg_dataset.gt_boxes[:num_vg]
    vg_dataset.gt_classes = vg_dataset.gt_classes[:num_vg]
    vg_dataset.relationships = vg_dataset.relationships[:num_vg]

    # adjust the results: keep the 0~50 columns
    for prediction in vg_predictions:
        #pred_rel_scores = prediction.get_field('pred_rel_scores')[:, :num_vg_predicates]
        pred_rel_scores = prediction.get_field('pred_rel_scores')
        rel_scores, pred_rel_labels = pred_rel_scores[:, 1:].max(1)
        pred_rel_labels = pred_rel_labels + 1
        _, sorting_idx = torch.sort(rel_scores.view(-1), dim=0, descending=True)
        prediction.add_field('rel_pair_idxs', prediction.get_field('rel_pair_idxs')[sorting_idx])
        prediction.add_field('pred_rel_scores', pred_rel_scores[sorting_idx])
        prediction.add_field('pred_rel_labels', pred_rel_labels[sorting_idx])

    # vrd part
    vrd_predictions = predictions[num_vg:]
    vrd_dataset = copy.deepcopy(dataset)
    vrd_dataset.filenames = vrd_dataset.filenames[num_vg:]
    vrd_dataset.img_info = vrd_dataset.img_info[num_vg:]
    vrd_to_union_map = dataset.vrd_to_union_map
    union_to_vrd_map = dataset.union_to_vrd_map
    sorted_vrd_keys = sorted(list(vrd_to_union_map.keys()))

    vrd_dataset.ind_to_predicates = []
    for k in sorted_vrd_keys:
        vrd_dataset.ind_to_predicates.append(dataset.ind_to_predicates[vrd_to_union_map[k]])
    # adjust the groundtruth
    vrd_dataset.gt_boxes = vrd_dataset.gt_boxes[num_vg:]
    vrd_dataset.gt_classes = vrd_dataset.gt_classes[num_vg:]
    vrd_dataset.relationships = vrd_dataset.relationships[num_vg:]
    # for i in range(len(vrd_dataset.relationships)):
    #     for j in range(len(vrd_dataset.relationships[i])):
    #         vrd_dataset.relationships[i][j, 2] = union_to_vrd_map[vrd_dataset.relationships[i][j, 2]]
    for prediction in vrd_predictions:
        # select the columns
        pred_rel_scores = prediction.get_field('pred_rel_scores')
        #column_idxs = [vrd_to_union_map[k] for k in sorted_vrd_keys]
        #pred_rel_scores = pred_rel_scores[:, column_idxs]
        rel_scores, pred_rel_labels = pred_rel_scores[:, 1:].max(1)
        pred_rel_labels = pred_rel_labels + 1
        _, sorting_idx = torch.sort(rel_scores.view(-1), dim=0, descending=True)
        prediction.add_field('rel_pair_idxs', prediction.get_field('rel_pair_idxs')[sorting_idx])
        prediction.add_field('pred_rel_scores', pred_rel_scores[sorting_idx])
        prediction.add_field('pred_rel_labels', pred_rel_labels[sorting_idx])

    vrd_folder, vg_folder = None, None
    if output_folder:
        vg_folder = os.path.join(output_folder, 'vg')
        mkdir(vg_folder)
        vrd_folder = os.path.join(output_folder, 'vrd')
        mkdir(vrd_folder)

    do_vrd_evaluation(
        cfg=cfg,
        dataset=vrd_dataset,
        predictions=vrd_predictions,
        output_folder=vrd_folder,
        logger=logger,
        iou_types=iou_types,
    )

    return do_vg_evaluation(
        cfg=cfg,
        dataset=vg_dataset,
        predictions=vg_predictions,
        output_folder=vg_folder,
        logger=logger,
        iou_types=iou_types,
    )
