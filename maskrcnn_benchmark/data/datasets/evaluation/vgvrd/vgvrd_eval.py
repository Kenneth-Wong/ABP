# ---------------------------------------------------------------
# vgvrd_eval.py
# Set-up time: 2022/2/28 17:23
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------
import logging
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, \
    SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
from maskrcnn_benchmark.data.datasets.visual_genome import HEAD, TAIL, BODY

eval_times = 0


def do_vgvrd_evaluation(
        cfg,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = len(dataset.ind_to_predicates)  # cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))

        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

    save_output(output_folder, groundtruths, predictions, dataset)

    avg_metrics = 0
    result_str = '\n' + '=' * 100 + '\n'
    result_dict_list_to_log = []

    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist()  # integer
            boxes = gt.bbox.tolist()  # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
            ],
            'annotations': anns,
        }
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id] * len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
            )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_res = COCOResults('bbox')
        coco_res.update(coco_eval)
        mAp = coco_eval.stats[1]

        def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
            p = coco_eval.params

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            if maxDets == -1:
                max_range_i = np.argmax(p.maxDets)
                mind = [max_range_i, ]
            else:
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if eval_type == 'precision':
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            elif eval_type == 'recall':
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            else:
                raise ValueError("Invalid eval metrics")
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return p.maxDets[mind[-1]], mean_s

        coco_res_to_save = {}
        for key, value in coco_res.results.items():
            for evl_name, eval_val in value.items():
                coco_res_to_save[f"{key}/{evl_name}"] = eval_val
        result_dict_list_to_log.append(coco_res_to_save)

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, 'recall')
        result_str += '=' * 100 + '\n'
        avg_metrics = mAp
        logger.info(result_str)
        logger.info("box evaluation done!")

    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # prepare all inputs
        global_container = {}
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes

        for groundtruth, prediction in zip(groundtruths, predictions):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)

        def generate_eval_res_dict(evaluator, mode):
            res_dict = {}
            for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
                res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
            return res_dict

        def longtail_part_eval(evaluator, mode):
            longtail_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
            assert "mean_recall" in evaluator.type
            res_dict = {}
            res_str = "\nlongtail part recall:\n"
            for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
                part_recall = {"h": [], "b": [], "t": [], }
                for idx, each_cat_recall in enumerate(cate_rec_list):
                    part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)
                res_dict[f"sgdet_longtail_part_recall/top{topk}/head"] = np.mean(part_recall['h'])
                res_dict[f"sgdet_longtail_part_recall/top{topk}/body"] = np.mean(part_recall['b'])
                res_dict[f"sgdet_longtail_part_recall/top{topk}/tail"] = np.mean(part_recall['t'])
                res_str += f"Top{topk:4}: head: {np.mean(part_recall['h']):.4f} " \
                           f"body: {np.mean(part_recall['b']):.4f} " \
                           f"tail: {np.mean(part_recall['t']):.4f}\n"

            return res_dict, res_str

        # show the distribution & recall_count
        pred_counter_dir = os.path.join(cfg.OUTPUT_DIR, "pred_counter.pkl")
        if os.path.exists(pred_counter_dir):
            with open(pred_counter_dir, 'rb') as f:
                pred_counter = pickle.load(f)

            def show_per_cls_performance_and_frequency(mean_recall_evaluator, per_cls_res_dict):
                cls_dict = mean_recall_evaluator.rel_name_list
                cate_recall = []
                cate_num = []
                cate_set = []
                counter_name = []
                for cate_set_idx, name_set in enumerate([HEAD, BODY, TAIL]):
                    for cate_id in name_set:
                        cate_set.append(cate_set_idx)
                        counter_name.append(cls_dict[cate_id - 1])  # list start from 0
                        cate_recall.append(per_cls_res_dict[cate_id - 1])  # list start from 0
                        cate_num.append(pred_counter[cate_id])  # dict start from 1

                def min_max_norm(data):
                    return (data - min(data)) / max(data)

                cate_num = min_max_norm(np.array(cate_num))
                cate_recall = np.array(cate_recall)
                # cate_recall = min_max_norm(np.array(cate_recall))

                fig, axs_c = plt.subplots(1, 1, figsize=(13, 5), tight_layout=True)
                pallte = ['r', 'g', 'b']
                color = [pallte[idx] for idx in cate_set]
                axs_c.bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
                axs_c.scatter(counter_name, cate_recall, color='k', zorder=10)

                plt.xticks(rotation=-90, )
                axs_c.grid()
                fig.set_facecolor((1, 1, 1))

                global eval_times

                save_file = os.path.join(cfg.OUTPUT_DIR,
                                         f"rel_freq_dist2recall-{mean_recall_evaluator.type}-{eval_times}.png")
                fig.savefig(save_file, dpi=300)

            per_cls_res_dict = eval_mean_recall.result_dict[f'{mode}_{eval_mean_recall.type}_list'][100]
            show_per_cls_performance_and_frequency(eval_mean_recall, per_cls_res_dict)

            global eval_times
            eval_times += 1

        longtail_part_res_dict, longtail_part_res_str = longtail_part_eval(eval_mean_recall, mode)

        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)

        result_str += longtail_part_res_str

        result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),

                                        generate_eval_res_dict(eval_mean_recall, mode)])

        result_str += '=' * 100 + '\n'

        # average the all recall and mean recall with the weight
        avg_metrics = np.mean(result_dict[mode + '_recall'][100]) * 0.5 \
                      + np.mean(result_dict[mode + '_mean_recall'][100]) * 0.5

        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))

    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'w') as f:
            f.write(result_str)

    return float(avg_metrics)


def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths': groundtruths, 'predictions': predictions},
                   os.path.join(output_folder, "eval_results.pytorch"))

        # with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths, predictions)):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
            ]
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]]  # xyxy, str
                for b, l in zip(prediction.bbox.tolist(), prediction.get_field('pred_labels').tolist())
            ]
            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
            })
        with open(os.path.join(output_folder, "visual_info.json"), "w") as f:
            json.dump(visual_info, f)


def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    # unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()  # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()  # (#gt_objs, )

    # about relations
    local_container['pred_rel_inds'] = prediction.get_field(
        'rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    local_container['rel_scores'] = prediction.get_field(
        'pred_rel_scores').detach().cpu().numpy()  # (#pred_rels, num_pred_class)

    # about objects
    local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()  # (#pred_objs, 4)
    local_container['pred_classes'] = prediction.get_field(
        'pred_labels').long().detach().cpu().numpy()  # (#pred_objs, )
    local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#pred_objs, )

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
    # for sgcls and predcls
    if mode != 'sgdet' and evaluator.get('eval_pair_accuracy') is not None:
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    if evaluator.get('eval_zeroshot_recall') is not None:
        evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if evaluator.get('eval_ng_zeroshot_recall') is not None:
        evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)

    if evaluator.get('eval_recall') is not None:
        local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    if evaluator.get('eval_nog_recall') is not None:
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    if evaluator.get('eval_pair_accuracy') is not None:
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    if evaluator.get('eval_mean_recall') is not None:
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    if evaluator.get('eval_ng_mean_recall') is not None:
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    if evaluator.get('eval_zeroshot_recall') is not None:
        evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # No Graph Constraint Zero-Shot Recall
    if evaluator.get('eval_ng_zeroshot_recall') is not None:
        evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

    return


def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets)  # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
    """
    from list of attribute indexs to [1,0,1,0,...,0,1] form
    """
    max_att = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attri_idx = (attributes.sum(-1) > 0).long()
    without_attri_idx = 1 - with_attri_idx
    num_pos = int(with_attri_idx.sum())
    num_neg = int(without_attri_idx.sum())
    assert num_pos + num_neg == num_obj

    attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

    for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
        for k in range(max_att):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets

