# ---------------------------------------------------------------
# open_image.py
# Set-up time: 2022/1/16 下午10:14
# Copyright (c) 2020 ICT
# Licensed under The MIT License [see LICENSE for details]
# Written by Kenneth-Wong (Wenbin-Wang) @ VIPL.ICT
# Contact: wenbin.wang@vipl.ict.ac.cn [OR] nkwangwenbin@gmail.com
# ---------------------------------------------------------------


import json
import logging
import os
import pickle
import random
from collections import defaultdict, OrderedDict, Counter

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.visual_genome import resampling_dict_generation, box_filter, \
    apply_resampling, load_windows
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, boxlist_union_area
from maskrcnn_benchmark.utils.comm import get_rank, synchronize


def load_cate_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    ind_to_predicates_cate = ['__background__'] + info['rel']
    ind_to_entites_cate = ['__background__'] + info['obj']

    # print(len(ind_to_predicates_cate))
    # print(len(ind_to_entites_cate))
    predicate_to_ind = {idx: name for idx, name in enumerate(ind_to_predicates_cate)}
    entites_cate_to_ind = {idx: name for idx, name in enumerate(ind_to_entites_cate)}

    return (ind_to_entites_cate, ind_to_predicates_cate,
            entites_cate_to_ind, predicate_to_ind)


def load_annotations(annotation_file, img_dir, num_img, split,
                     filter_empty_rels, ):
    """

    :param annotation_file:
    :param img_dir:
    :param img_range:
    :param filter_empty_rels:
    :return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    annotations = json.load(open(annotation_file, 'r'))

    if num_img == -1:
        num_img = len(annotations)

    annotations = annotations[:num_img]

    empty_list = set()
    if filter_empty_rels:
        for i, each in enumerate(annotations):
            if len(each['rel']) == 0:
                empty_list.add(i)
            if len(each['bbox']) == 0:
                empty_list.add(i)

    print('empty relationship image num: ', len(empty_list))

    boxes = []
    gt_classes = []
    relationships = []
    img_info = []
    for i, anno in enumerate(annotations):

        if i in empty_list:
            continue

        boxes_i = np.array(anno['bbox'])
        gt_classes_i = np.array(anno['det_labels'], dtype=int)

        rels = np.array(anno['rel'], dtype=int)

        gt_classes_i += 1
        rels[:, -1] += 1

        image_info = {
            'width': anno['img_size'][0],
            'height': anno['img_size'][1],
            'img_fn': os.path.join(img_dir, anno['img_fn'] + '.jpg')
        }

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        img_info.append(image_info)

    return boxes, gt_classes, relationships, img_info


class OIDataset(torch.utils.data.Dataset):


    def __init__(self, split, img_dir, ann_file, cate_info_file, transforms=None,
                 num_im=-1, check_img_file=False, filter_duplicate_rels=True, flip_aug=False,
                 window_file=None):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        self.HEAD = []
        self.BODY = []
        self.TAIL = []

        for i, cate in enumerate(cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT):
            if cate == 'h':
                self.HEAD.append(i)
            elif cate == 'b':
                self.BODY.append(i)
            elif cate == 't':
                self.TAIL.append(i)

        # for debug
        if cfg.DEBUG:
            num_im = 5400 if split == "train" else 10  # 2290

        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split
        self.img_dir = img_dir
        self.cate_info_file = cate_info_file
        self.annotation_file = ann_file
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.check_img_file = check_img_file
        self.remove_tail_classes = False

        self.mtl_window = cfg.MODEL.MTL_WINDOW

        (self.ind_to_classes,
         self.ind_to_predicates,
         self.classes_to_ind,
         self.predicates_to_ind) = load_cate_info(self.cate_info_file)  # contiguous 151, 51 containing __background__

        logger = logging.getLogger("maskrcnn_benchmark.dataset")
        self.logger = logger

        self.categories = {i: self.ind_to_classes[i]
                           for i in range(len(self.ind_to_classes))}

        self.gt_boxes, self.gt_classes, self.relationships, self.img_info = load_annotations(
            self.annotation_file, img_dir, num_im, split=split,
            filter_empty_rels=False if not cfg.MODEL.RELATION_ON and split == "train" else True,
        )

        self.filenames = [img_if['img_fn'] for img_if in self.img_info]
        self.idx_list = list(range(len(self.filenames)))

        self.id_to_img_map = {k: v for k, v in enumerate(self.idx_list)}

        # mtl window
        self.window_file = None
        if self.mtl_window and self.split == "train":
            assert window_file is not None
            self.window_file = window_file
            self.windows, self.head_bboxes, self.tail_bboxes, \
            self.clipped_head_bboxes, self.clipped_tail_bboxes, \
            self.norm_clipped_head_bboxes, self.norm_clipped_tail_bboxes, self.relation_triplets, \
            self.referred_nums, self.referred_indices = load_windows(self.window_file,
                                                                     np.ones(len(self.filenames)).astype(bool),
                                                                     len(self.ind_to_classes))

        if cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING and self.split == 'train':
            self.resampling_method = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_METHOD
            assert self.resampling_method in ['bilvl', 'lvis']

            self.global_rf = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR
            self.drop_rate = cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE
            # creat repeat dict in main process, other process just wait and load
            if get_rank() == 0:
                repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)
                self.repeat_dict = repeat_dict
                with open(os.path.join(cfg.OUTPUT_DIR, "repeat_dict.pkl"), "wb") as f:
                    pickle.dump(self.repeat_dict, f)

            synchronize()
            self.repeat_dict = resampling_dict_generation(self, self.ind_to_predicates, logger)

            duplicate_idx_list = []
            for idx in range(len(self.filenames)):
                r_c = self.repeat_dict[idx]
                duplicate_idx_list.extend([idx for _ in range(r_c)])
            self.idx_list = duplicate_idx_list

    def __getitem__(self, index):
        # if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))
        if self.repeat_dict is not None:
            index = self.idx_list[index]

        img = Image.open(self.filenames[index]).convert("RGB")
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                  ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

        target = self.get_groundtruth(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_statistics(self):

        fg_matrix, bg_matrix, rel_counter_init, cooccur_matrix, occur_num = get_OI_statistics(img_dir=self.img_dir,
                                                                                              ann_file=self.annotation_file,
                                                                                              cate_info_file=self.cate_info_file,
                                                                                              must_overlap=True,
                                                                                              window_file=self.window_file)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = fg_matrix / fg_matrix.sum(2)[:, :, None] + eps

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_classes[:cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES],  # fake
            'cooccur_matrix': torch.from_numpy(cooccur_matrix).float(),
            'occur_num': torch.from_numpy(occur_num).float()
        }

        rel_counter = Counter()

        for i in tqdm(self.idx_list):
            relation = self.relationships[i].copy()  # (num_rel, 3)
            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert self.split == 'train'
                old_size = relation.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in relation:
                    all_rel_sets[(o0, o1)].append(r)
                relation = [(k[0], k[1], np.random.choice(v))
                            for k, v in all_rel_sets.items()]
                relation = np.array(relation, dtype=np.int32)

            if self.repeat_dict is not None:
                relation, _ = apply_resampling(i, relation, self.repeat_dict, self.drop_rate)

            for i in relation[:, -1]:
                if i > 0:
                    rel_counter[i] += 1

        cate_num = []
        cate_num_init = []
        cate_set = []
        counter_name = []

        sorted_cate_list = [i[0] for i in rel_counter_init.most_common()]
        lt_part_dict = cfg.MODEL.ROI_RELATION_HEAD.LONGTAIL_PART_DICT
        for cate_id in sorted_cate_list:
            if lt_part_dict[cate_id] == 'h':
                cate_set.append(0)
            if lt_part_dict[cate_id] == 'b':
                cate_set.append(1)
            if lt_part_dict[cate_id] == 't':
                cate_set.append(2)

            counter_name.append(self.ind_to_predicates[cate_id])  # list start from 0
            cate_num.append(rel_counter[cate_id])  # dict start from 1
            cate_num_init.append(rel_counter_init[cate_id])  # dict start from 1

        pallte = ['r', 'g', 'b']
        color = [pallte[idx] for idx in cate_set]

        fig, axs_c = plt.subplots(3, 1, figsize=(16, 15), tight_layout=True)
        fig.set_facecolor((1, 1, 1))

        axs_c[0].bar(counter_name, cate_num_init, color=color, width=0.6, zorder=0)
        axs_c[0].grid()
        axs_c[0].set_title("distribution of original data")
        plt.sca(axs_c[0])
        plt.xticks(rotation=-60, )

        axs_c[1].bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
        axs_c[1].grid()
        axs_c[1].set_title("distribution of training data")
        axs_c[1].set_ylim(0, 50000)
        plt.sca(axs_c[1])
        plt.xticks(rotation=-60, )

        ##############################################################################
        # transfer from data/build.py: get_dataset_distribution function, draw predicates in h/b/t group
        with open(os.path.join(cfg.OUTPUT_DIR, "pred_counter.pkl"), 'wb') as f:
            pickle.dump(rel_counter, f)

        count_sorted = []
        counter_name = []
        cate_set = []
        cls_dict = self.ind_to_predicates

        for idx, name_set in enumerate([self.HEAD, self.BODY, self.TAIL]):
            # sort the cate names accoding to the frequency
            part_counter = []
            for name in name_set:
                part_counter.append(rel_counter[name])
            part_counter = np.array(part_counter)
            sorted_idx = np.flip(np.argsort(part_counter))

            # reaccumulate the frequency in sorted index
            for j in sorted_idx:
                name = name_set[j]
                cate_set.append(idx)
                counter_name.append(cls_dict[name])
                count_sorted.append(rel_counter[name])

        count_sorted = np.array(count_sorted)

        palate = ['r', 'g', 'b']
        color = [palate[idx] for idx in cate_set]
        axs_c[2].bar(counter_name, count_sorted, color=color, width=0.6, zorder=0)
        axs_c[2].grid()
        title = "distribution of training data (hbt)"
        if self.repeat_dict is not None:
            title += " (balance sampled)"
        axs_c[2].set_title(title)
        axs_c[2].set_ylim(0, 50000)
        plt.sca(axs_c[2])
        plt.xticks(rotation=-60)

        save_file = os.path.join(cfg.OUTPUT_DIR, "rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)

        return result

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, need_window=True, inner_idx=True):
        if not inner_idx:
            # here, if we pass the index after resampeling, we need to map back to the initial index
            if self.repeat_dict is not None:
                index = self.idx_list[index]

        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        box = self.gt_boxes[index]
        box = torch.from_numpy(box)  # guard against no boxes

        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(np.zeros((len(self.gt_classes[index]), 10))))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        relation_non_masked = None
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index,
                                                             relation,
                                                             self.repeat_dict,
                                                             self.drop_rate, )
        # add relation to target
        num_box = len(target)
        relation_map_non_masked = None
        if self.repeat_dict is not None:
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)

        relation_map = torch.zeros((num_box, num_box), dtype=torch.long)
        for i in range(relation.shape[0]):
            # Sometimes two objects may have multiple different ground-truth predicates in VisualGenome.
            # In this case, when we construct GT annotations, random selection allows later predicates
            # having the chance to overwrite the precious collided predicate.
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] != 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                    if relation_map_non_masked is not None:
                        relation_map_non_masked[int(relation_non_masked[i, 0]),
                                                int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
                if relation_map_non_masked is not None:
                    relation_map_non_masked[int(relation_non_masked[i, 0]),
                                            int(relation_non_masked[i, 1])] = int(relation_non_masked[i, 2])

        target.add_field("relation", relation_map, is_triplet=True)
        if relation_map_non_masked is not None:
            target.add_field("relation_non_masked", relation_map_non_masked.long(), is_triplet=True)

        # add windows
        if self.mtl_window and need_window and self.split == "train":
            window_target = self.process_windows(index, (w, h))
            target.add_field("window", window_target)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def process_windows(self, index, size):
        windows = self.windows[index].copy()

        # for images that being failed sampling
        if windows.shape[0] == 0:
            window_target = BoxList(windows, size, 'xyxy')
            window_target.add_field("head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("relation", torch.zeros(0, 3).long())
            window_target.add_field("num", torch.zeros(0, ).long())
            window_target.add_field("window_label", torch.zeros(0, len(self.ind_to_predicates)).float())
            return window_target

        # sub_target = BoxList(windows, (w, h), 'xyxy')
        head_bboxes, tail_bboxes = self.head_bboxes[index].copy(), self.tail_bboxes[index].copy()
        clipped_head_bboxes, clipped_tail_bboxes = self.clipped_head_bboxes[index].copy(), \
                                                   self.clipped_tail_bboxes[index].copy()
        norm_clipped_head_bboxes, norm_clipped_tail_bboxes = self.norm_clipped_head_bboxes[index].copy(), \
                                                             self.norm_clipped_tail_bboxes[index].copy()
        referred_nums = self.referred_nums[index].copy()
        referred_indices = self.referred_indices[index].copy()
        relation = self.relation_triplets[index].copy()

        # if resampled is applied:
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index, relation, self.repeat_dict, self.drop_rate)

        # check each window
        win_referred_indices = np.split(referred_indices, np.cumsum(referred_nums))[:-1]
        win_referred_relations = [relation[indices] for indices in win_referred_indices]
        win_head_bboxes = np.split(head_bboxes, np.cumsum(referred_nums))[:-1]
        win_tail_bboxes = np.split(tail_bboxes, np.cumsum(referred_nums))[:-1]
        win_clipped_head_bboxes = np.split(clipped_head_bboxes, np.cumsum(referred_nums))[:-1]
        win_clipped_tail_bboxes = np.split(clipped_tail_bboxes, np.cumsum(referred_nums))[:-1]
        win_norm_clipped_head_bboxes = np.split(norm_clipped_head_bboxes, np.cumsum(referred_nums))[:-1]
        win_norm_clipped_tail_bboxes = np.split(norm_clipped_tail_bboxes, np.cumsum(referred_nums))[:-1]

        window_num = cfg.MODEL.WINDOW_NUM
        max_rel_num = cfg.MODEL.MAX_REL_NUM
        label_type = cfg.MODEL.WINDOW_LABEL_TYPE
        all_windows = []
        all_rels = []
        all_head_bboxes, all_tail_bboxes, all_clipped_head_bboxes, all_clipped_tail_bboxes = [], [], [], []
        nums = []
        all_win_label = []
        randperm = list(range(len(windows)))
        np.random.shuffle(randperm)
        for win_idx in randperm:
            window_i = windows[win_idx]
            rel_i = win_referred_relations[win_idx]
            hb_i = win_head_bboxes[win_idx]
            tb_i = win_tail_bboxes[win_idx]
            chb_i = win_clipped_head_bboxes[win_idx]
            ctb_i = win_clipped_tail_bboxes[win_idx]
            nchb_i = win_norm_clipped_head_bboxes[win_idx]
            nctb_i = win_norm_clipped_tail_bboxes[win_idx]

            if len(all_windows) >= window_num:
                break
            if rel_i.ndim < 2:
                print(index, win_referred_relations)
            if len(np.where(rel_i[:, -1] > 0)[0]) == 0:
                continue

            rel_idx = np.where(rel_i[:, -1] > 0)[0]
            if len(rel_idx) > max_rel_num:
                selected = np.sort(np.random.choice(list(range(len(rel_idx))), max_rel_num, replace=False))
                rel_idx = rel_idx[selected]
            rel_i = rel_i[rel_idx]
            hb_i = hb_i[rel_idx]
            tb_i = tb_i[rel_idx]
            chb_i, ctb_i = chb_i[rel_idx], ctb_i[rel_idx]
            nchb_i, nctb_i = nchb_i[rel_idx], nctb_i[rel_idx]

            soft_label = np.zeros(len(self.ind_to_predicates), dtype=np.float32)
            if label_type == 'hard':
                soft_label[np.unique(rel_i[:, -1])] = 1.
            elif label_type == 'soft':
                ncb_i = cat_boxlist([BoxList(nchb_i, size, 'xyxy'), BoxList(nctb_i, size, 'xyxy')])
                area_total = boxlist_union_area(ncb_i)
                soft_label[0] = np.sqrt(max(0., 1 - area_total))
                for predicate in np.unique(rel_i[:, -1]):
                    cls_indices = np.where(rel_i[:, -1] == predicate)[0]
                    cls_ncb_i = cat_boxlist([BoxList(nchb_i, size, 'xyxy')[cls_indices],
                                             BoxList(nctb_i, size, 'xyxy')[cls_indices]])
                    area_total = boxlist_union_area(cls_ncb_i)
                    soft_label[predicate] = np.sqrt(area_total)
            else:
                raise NotImplementedError
            soft_label = soft_label / np.sum(soft_label)

            all_windows.append(window_i[None])
            all_head_bboxes.append(hb_i)
            all_tail_bboxes.append(tb_i)
            all_clipped_head_bboxes.append(chb_i)
            all_clipped_tail_bboxes.append(ctb_i)
            all_rels.append(rel_i)
            all_win_label.append(soft_label[None])
            nums.append(hb_i.shape[0])

        if len(all_windows) == 0:
            window_target = BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy')
            window_target.add_field("head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("relation", torch.zeros(0, 3).long())
            window_target.add_field("num", torch.zeros(0, ).long())
            window_target.add_field("window_label", torch.zeros(0, len(self.ind_to_predicates)).float())
            return window_target

        window_target = BoxList(np.vstack(all_windows), size, 'xyxy')
        window_target.add_field("head_bbox", BoxList(np.vstack(all_head_bboxes), size, 'xyxy'))
        window_target.add_field("tail_bbox", BoxList(np.vstack(all_tail_bboxes), size, 'xyxy'))
        window_target.add_field("clipped_head_bbox", BoxList(np.vstack(all_clipped_head_bboxes), size, 'xyxy'))
        window_target.add_field("clipped_tail_bbox", BoxList(np.vstack(all_clipped_tail_bboxes), size, 'xyxy'))
        window_target.add_field("relation", torch.from_numpy(np.vstack(all_rels)).long())
        window_target.add_field("num", torch.from_numpy(np.array(nums)).long())
        window_target.add_field("window_label", torch.from_numpy(np.vstack(all_win_label)).float())
        return window_target

    def __len__(self):
        return len(self.idx_list)


def get_OI_statistics(img_dir, ann_file, cate_info_file, must_overlap=True, window_file=None):
    train_data = OIDataset(split='train', img_dir=img_dir, ann_file=ann_file, cate_info_file=cate_info_file,
                           filter_duplicate_rels=False, window_file=window_file)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    cooccur_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
    occur_num = np.zeros(num_obj_classes, dtype=np.int64)
    rel_counter = Counter()

    for ex_ind in tqdm(range(len(train_data.img_info))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
            rel_counter[gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

        unique_classes = np.unique(gt_classes)
        for i in range(len(unique_classes)):
            occur_num[unique_classes[i]] += 1
            for j in range(i + 1, len(unique_classes)):
                cooccur_matrix[unique_classes[i], unique_classes[j]] += 1
                cooccur_matrix[unique_classes[j], unique_classes[i]] += 1
    return fg_matrix, bg_matrix, rel_counter, cooccur_matrix, occur_num
