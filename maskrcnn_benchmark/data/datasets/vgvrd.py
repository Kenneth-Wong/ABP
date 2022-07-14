import os
import sys
import torch
import h5py
import json
import logging
from PIL import Image
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from tqdm import tqdm
import random
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')  # forbid the pop window
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, cat_boxlist, boxlist_union_area
from maskrcnn_benchmark.utils.comm import get_rank, synchronize
from maskrcnn_benchmark.data.datasets.bi_lvl_rsmp import resampling_dict_generation, apply_resampling
from maskrcnn_benchmark.data.datasets.vrd import load_graphs as load_vrd_graphs
from maskrcnn_benchmark.data.datasets.vrd import load_image_infos as load_vrd_image_infos
from maskrcnn_benchmark.data.datasets.visual_genome import load_windows
import pickle

BOX_SCALE = 1024  # Scale at which we have the boxes


class VGVRDDataset(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 img_dir,
                 roidb_file,
                 dict_file,
                 image_file, transforms=None,
                 window_file=None,
                 num_im=-1, num_val_im=5000,
                 filter_empty_rels=True,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 ):

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

        num_vrd_im = -1

        # for debug
        if cfg.DEBUG:
            num_im = 5400 if split == 'train' else 2290
            num_val_im = 2 #400
            num_vrd_im = 10

        assert split in {'train', 'val', 'test'}
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.repeat_dict = None
        self.mtl_window = cfg.MODEL.MTL_WINDOW

        self.ind_to_classes, self.ind_to_predicates, \
        self.union_to_vrd_map, self.vrd_to_union_map, self.vrd_to_vg_map = load_info(
            dict_file)  # contiguous 151, 51 containing __background__
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        logger = logging.getLogger("maskrcnn_benchmark.dataset")
        self.logger = logger

        # load VG part
        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file[0], self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
        )

        self.filenames, self.img_info = load_image_filenames(img_dir[0], image_file[0])  # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

        # load VRD part
        vrd_img_ids, vrd_gt_boxes, vrd_gt_classes, vrd_relationships = load_vrd_graphs(
            self.roidb_file[1], num_vrd_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=False,
        )
        vrd_img_infos = load_vrd_image_infos(image_file[1])
        vrd_img_infos = [vrd_img_infos[id] for id in vrd_img_ids]
        vrd_filenames = [os.path.join(img_dir[1], img_info['filename']) for img_info in vrd_img_infos]

        # concatenate two datasets
        self.gt_boxes += vrd_gt_boxes
        self.gt_classes += vrd_gt_classes
        self.relationships += vrd_relationships
        self.filenames += vrd_filenames
        self.img_info += vrd_img_infos

        self.idx_list = list(range(len(self.filenames)))

        # use once
        # predicate_counter = np.zeros(len(self.ind_to_predicates))
        # for rels in self.relationships:
        #     for rel in rels[:, 2]:
        #         predicate_counter[rel] += 1
        # lbt_dict = []
        # for i in predicate_counter:
        #     if i > 10000:
        #         lbt_dict.append('h')
        #     elif i > 500:
        #         lbt_dict.append('b')
        #     else:
        #         lbt_dict.append('t')
        # print(lbt_dict)

        # mtl window
        self.window_file = None
        if self.mtl_window and self.split == "train":
            assert window_file is not None
            self.window_file = window_file
            # vg part
            self.windows, self.head_bboxes, self.tail_bboxes, \
            self.clipped_head_bboxes, self.clipped_tail_bboxes, \
            self.norm_clipped_head_bboxes, self.norm_clipped_tail_bboxes, self.relation_triplets, \
            self.referred_nums, self.referred_indices = load_windows(self.window_file[0], self.split_mask,
                                                                     len(self.ind_to_classes))

            # vrd part
            vrd_windows, vrd_head_bboxes, vrd_tail_bboxes, \
            vrd_clipped_head_bboxes, vrd_clipped_tail_bboxes, \
            vrd_norm_clipped_head_bboxes, vrd_norm_clipped_tail_bboxes, vrd_relation_triplets, \
            vrd_referred_nums, vrd_referred_indices = load_windows(self.window_file[1],
                                                                   np.ones(len(vrd_filenames)).astype(bool),
                                                                   len(self.ind_to_classes))

            # concate
            self.windows += vrd_windows
            self.head_bboxes += vrd_head_bboxes
            self.tail_bboxes += vrd_tail_bboxes
            self.clipped_head_bboxes += vrd_clipped_head_bboxes
            self.clipped_tail_bboxes += vrd_clipped_tail_bboxes
            self.norm_clipped_head_bboxes += vrd_norm_clipped_head_bboxes
            self.norm_clipped_tail_bboxes += vrd_norm_clipped_tail_bboxes
            self.relation_triplets += vrd_relation_triplets
            self.referred_nums += vrd_referred_nums
            self.referred_indices += vrd_referred_indices

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
        fg_matrix, bg_matrix, rel_counter_init = get_VGVRD_statistics(img_dir=self.img_dir,
                                                                      roidb_file=self.roidb_file,
                                                                      dict_file=self.dict_file,
                                                                      image_file=self.image_file,
                                                                      must_overlap=True,
                                                                      window_file=self.window_file)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_classes[:cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES],
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
                relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
                relation = np.array(relation, dtype=np.int32)

            if self.repeat_dict is not None:
                relation, _ = apply_resampling(i, relation, self.repeat_dict, self.drop_rate, )

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

        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']

        if 'coco_id' in img_info: # image from vg
            box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        else:
            box = self.gt_boxes[index]

        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes

        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_classes[index]))

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

        # add relation to target
        num_box = len(target)
        relation_map_non_masked, relation_non_masked = None, None
        if self.repeat_dict is not None:
            relation, relation_non_masked = apply_resampling(index, relation, self.repeat_dict, self.drop_rate)
            relation_map_non_masked = torch.zeros((num_box, num_box), dtype=torch.long)

        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
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

        # add image level labels
        # img_lvl_obj_label = np.zeros(len(self.ind_to_classes), dtype=np.float32)
        # img_lvl_rel_label = np.zeros(len(self.ind_to_predicates), dtype=np.float32)
        # img_lvl_rel_label[relation[:, -1]] = 1
        # counter = Counter(self.gt_classes[index])
        # for k, v in counter.items():
        #     if k > 0:
        #         img_lvl_obj_label[k] = v / len(self.gt_classes[index])
        # img_lvl_obj_label = (img_lvl_obj_label + 1e-3) / np.sum(img_lvl_obj_label + 1e-3)
        # counter = Counter(relation[:, -1])
        # for k, v in counter.items():
        #     if k > 0:
        #         img_lvl_rel_label[k] = v / len(relation)
        # img_lvl_rel_label = (img_lvl_rel_label + 1e-3) / np.sum(img_lvl_rel_label + 1e-3)
        # target.add_field("img_lvl_obj_label", torch.from_numpy(img_lvl_obj_label).float(), is_global=True)
        # target.add_field("img_lvl_rel_label", torch.from_numpy(img_lvl_rel_label).float(), is_global=True)

        # add windows
        if self.mtl_window and need_window and self.split == "train":
            window_target = self.process_windows(index, (w, h))
            target.add_field("window", window_target)

        # global_relation_label = np.zeros(len(self.ind_to_predicates), dtype=np.float32)
        # if len(np.where(relation[:, -1] > 0)[0]) > 0:
        #     global_relation_label[relation[:, -1][np.where(relation[:, -1] > 0)[0]]] = 1
        # global_relation_label = global_relation_label + 1e-3
        # global_relation_label = np.log(global_relation_label / global_relation_label.sum())[None]
        # target.add_field("global_relation_label", torch.from_numpy(global_relation_label).repeat(len(box), 1).float())

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
        else:
            target = target.clip_to_image(remove_empty=True)

        return target

    def process_windows(self, index, size):
        windows = self.windows[index].copy()

        # for images that being failed sampling
        if windows.shape[0] == 0:
            window_target = BoxList(windows, size, 'xyxy')
            # window_target.add_field("object_label", torch.zeros(0, len(self.ind_to_classes)).float())
            window_target.add_field("head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("relation", torch.zeros(0, 3).long())
            window_target.add_field("num", torch.zeros(0, ).long())
            window_target.add_field("window_label", torch.zeros(0, len(self.ind_to_predicates)).float())
            return window_target

        # sub_target = BoxList(windows, (w, h), 'xyxy')
        # object_labels = self.soft_object_labels[index].copy()
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
        # all_object_labels = []
        all_rels = []
        all_head_bboxes, all_tail_bboxes, all_clipped_head_bboxes, all_clipped_tail_bboxes = [], [], [], []
        nums = []
        all_win_label = []
        randperm = list(range(len(windows)))
        np.random.shuffle(randperm)
        for win_idx in randperm:
            window_i = windows[win_idx]
            # object_label_i = object_labels[win_idx]
            rel_i = win_referred_relations[win_idx]
            hb_i = win_head_bboxes[win_idx]
            tb_i = win_tail_bboxes[win_idx]
            chb_i = win_clipped_head_bboxes[win_idx]
            ctb_i = win_clipped_tail_bboxes[win_idx]
            nchb_i = win_norm_clipped_head_bboxes[win_idx]
            nctb_i = win_norm_clipped_tail_bboxes[win_idx]
            # for window_i, rel_i, hb_i, tb_i, chb_i, ctb_i, nchb_i, nctb_i in zip(windows, win_referred_relations, win_head_bboxes,
            #                                                            win_tail_bboxes, win_clipped_head_bboxes,
            #                                                            win_clipped_tail_bboxes, win_norm_clipped_head_bboxes,
            #                                                            win_norm_clipped_tail_bboxes):
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
            # all_object_labels.append(object_label_i[None])
            all_head_bboxes.append(hb_i)
            all_tail_bboxes.append(tb_i)
            all_clipped_head_bboxes.append(chb_i)
            all_clipped_tail_bboxes.append(ctb_i)
            all_rels.append(rel_i)
            all_win_label.append(soft_label[None])
            nums.append(hb_i.shape[0])

        if len(all_windows) == 0:
            window_target = BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy')
            # window_target.add_field("object_label", torch.zeros(0, len(self.ind_to_classes)).float())
            window_target.add_field("head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_head_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("clipped_tail_bbox", BoxList(np.zeros((0, 4), dtype=np.float32), size, 'xyxy'))
            window_target.add_field("relation", torch.zeros(0, 3).long())
            window_target.add_field("num", torch.zeros(0, ).long())
            window_target.add_field("window_label", torch.zeros(0, len(self.ind_to_predicates)).float())
            return window_target

        window_target = BoxList(np.vstack(all_windows), size, 'xyxy')
        # window_target.add_field("object_label", torch.from_numpy(np.vstack(all_object_labels)).float())
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


def get_VGVRD_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True, window_file=None):
    train_data = VGVRDDataset(split='train', img_dir=img_dir, roidb_file=roidb_file,
                              dict_file=dict_file, image_file=image_file, num_val_im=5000,
                              filter_duplicate_rels=True, window_file=window_file)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)
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
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix, rel_counter


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file):
    """
    Loads the file containing the visual genome label meanings
    """
    # load vg
    info = json.load(open(dict_file, 'r'))

    ind_to_classes = info['objects']
    ind_to_predicates = info['predicates']
    union_to_vrd_map = {int(k): int(v) for k, v in info['self_to_vrd'].items()}
    vrd_to_union_map = {int(k): int(v) for k, v in info['vrd_to_self'].items()}
    vrd_to_vg_map = {int(k): int(v) for k, v in info['vrd_to_vg'].items()}
    return ind_to_classes, ind_to_predicates, union_to_vrd_map, vrd_to_union_map, vrd_to_vg_map


def load_image_filenames(img_dir, image_file):
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
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
