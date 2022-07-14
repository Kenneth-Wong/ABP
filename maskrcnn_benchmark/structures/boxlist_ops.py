# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import scipy.linalg

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode), keep


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_union(boxlist1, boxlist2):
    """
    Compute the union region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) union, sized [N,4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    union_box = torch.cat((
        torch.min(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.max(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    return BoxList(union_box, boxlist1.size, "xyxy")

def boxlist_intersection(boxlist1, boxlist2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) intersection, sized [N,4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat((
        torch.max(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.min(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    invalid_bbox = torch.max((inter_box[:,0] >= inter_box[:,2]).long(), (inter_box[:,1] >= inter_box[:,3]).long())
    inter_box[invalid_bbox > 0] = 0
    return BoxList(inter_box, boxlist1.size, "xyxy")

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if field in bboxes[0].triplet_extra_fields:
            triplet_list = [bbox.get_field(field).numpy() for bbox in bboxes]
            data = torch.from_numpy(scipy.linalg.block_diag(*triplet_list))
            cat_boxes.add_field(field, data, is_triplet=True)
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
            cat_boxes.add_field(field, data)

    return cat_boxes


def boxlist_clip_and_change_coordinate_to_window(head_bboxes, tail_bboxes, window, labels=None):
    """
        Clip the BoxList (having the same image size) into windows and change the coordinates relative to windows.
        Arguments:
            head_bboxes, tail_bboxes (BoxList)
            window torch.tensor, a single window
            labels, torch.tensor
    """
    w, h = head_bboxes.size
    hx1, hy1, hx2, hy2,  = head_bboxes._split_into_xyxy()
    tx1, ty1, tx2, ty2, = tail_bboxes._split_into_xyxy()

    wx1, wy1, wx2, wy2 = window[0]
    win_height, win_width = wy2 - wy1, wx2 - wx1

    hcx1 = torch.clamp(hx1, min=float(wx1), max=float(wx2))
    hcx2 = torch.clamp(hx2, min=float(wx1), max=float(wx2))
    hcy1 = torch.clamp(hy1, min=float(wy1), max=float(wy2))
    hcy2 = torch.clamp(hy2, min=float(wy1), max=float(wy2))
    head_clipped = BoxList(torch.cat([hcx1, hcy1, hcx2, hcy2], dim=1), (w, h), 'xyxy')
    head_areas = (head_clipped.bbox[:, 2] - head_clipped.bbox[:, 0]) * (head_clipped.bbox[:, 3] - head_clipped.bbox[:, 1])

    tcx1 = torch.clamp(tx1, min=float(wx1), max=float(wx2))
    tcx2 = torch.clamp(tx2, min=float(wx1), max=float(wx2))
    tcy1 = torch.clamp(ty1, min=float(wy1), max=float(wy2))
    tcy2 = torch.clamp(ty2, min=float(wy1), max=float(wy2))
    tail_clipped = BoxList(torch.cat([tcx1, tcy1, tcx2, tcy2], dim=1), (w, h), 'xyxy')
    tail_areas = (tail_clipped.bbox[:, 2] - tail_clipped.bbox[:, 0]) * (tail_clipped.bbox[:, 3] - tail_clipped.bbox[:, 1])

    # only keep the fg pair and non-masked pair (label==-1) and the overlap with the window > 0
    indices = (head_areas > 0) & (tail_areas > 0)
    if labels is not None:
        indices = indices & (labels > 0)
    indices = torch.nonzero(indices).view(-1)

    if len(indices) == 0:
        return None
    clipped_head_bboxes = head_clipped[indices]
    clipped_tail_bboxes = tail_clipped[indices]

    norm_clipped_head_bboxes = clipped_head_bboxes.copy()
    norm_clipped_tail_bboxes = clipped_tail_bboxes.copy()

    norm_clipped_head_bboxes.bbox = (norm_clipped_head_bboxes.bbox -
                                torch.tensor([wx1, wy1, wx1, wy1], device=window.device)[None]) / \
                                torch.tensor([win_width, win_height, win_width, win_height], device=window.device)[None]

    norm_clipped_tail_bboxes.bbox = (norm_clipped_tail_bboxes.bbox -
                                torch.tensor([wx1, wy1, wx1, wy1], device=window.device)[None])/ \
                                torch.tensor([win_width, win_height, win_width, win_height], device=window.device)[None]

    return clipped_head_bboxes, clipped_tail_bboxes, norm_clipped_head_bboxes, norm_clipped_tail_bboxes, indices



def boxlist_union_area(bboxes):
    """
        Get the union area of a set of boxes
        Arguments:
            bboxes (BoxList)
    """
    area_total = 0
    num_boxes = len(bboxes)
    w, h = bboxes.size
    res = []
    idx = [[i] for i in range(num_boxes)]
    comb_idx = idx.copy()
    original_bboxes = bboxes.copy()
    comb_bboxes = bboxes.copy()
    sign = 1
    while len(comb_bboxes):
        # NOTE: cannot use .area() which +1
        area = float(torch.sum((comb_bboxes.bbox[:, 2] - comb_bboxes.bbox[:, 0]) * (comb_bboxes.bbox[:, 3] - comb_bboxes.bbox[:, 1])))
        area_total += area * sign
        sign *= -1

        # compute the intersection box of the original bboxes and the comb bboxes
        lt = torch.max(original_bboxes.bbox[:, None, :2], comb_bboxes.bbox[None, :, :2])  # [N,M,2]
        rb = torch.min(original_bboxes.bbox[:, None, 2:], comb_bboxes.bbox[None, :, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        intersec_area = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        ltrb = torch.cat((lt, rb), dim=-1)  # [N, M, 4]

        # compute the combination
        tmp_idx = []
        tmp_comb_bboxes = []
        for i, id_i in enumerate(idx):
            for j, id_j in enumerate(comb_idx):
                if id_j[0] > id_i[0] and intersec_area[i, j] > 0:
                    n_id_j = id_j.copy()
                    n_id_j.insert(0, id_i[0])
                    tmp_idx.append(n_id_j)
                    tmp_comb_bboxes.append(ltrb[i, j][None])
        comb_idx = tmp_idx.copy()
        if len(tmp_comb_bboxes) == 0:
            comb_bboxes = BoxList(torch.zeros(0, 4), (w, h), 'xyxy')
        else:
            comb_bboxes = BoxList(torch.cat(tmp_comb_bboxes, 0), (w, h), 'xyxy')
        del ltrb, lt, rb, wh, intersec_area, tmp_idx
    return area_total
