from scipy.optimize import linear_sum_assignment
import numpy as np

def evaluate_segmentation(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """
    Evaluates instance segmentation quality between predicted and ground truth labels.
    Uses Hungarian matching to handle arbitrary label permutations.

    Metrics:
        - Panoptic Quality (PQ) = SQ * RQ
        - Segmentation Quality (SQ): mean IoU of matched pairs
        - Recognition Quality (RQ): F1 score of matched pairs
        - Coverage: fraction of GT points correctly assigned
    """
    pred_ids = np.unique(pred_labels)
    gt_ids   = np.unique(gt_labels)

    # build IoU matrix (gt x pred)
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    for i, g in enumerate(gt_ids):
        gt_mask = gt_labels == g
        for j, p in enumerate(pred_ids):
            pred_mask  = pred_labels == p
            intersection = (gt_mask & pred_mask).sum()
            if intersection == 0:
                continue
            union = (gt_mask | pred_mask).sum()
            iou_matrix[i, j] = intersection / union

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    iou_threshold = 0.5
    matched_iou = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_iou.append(iou_matrix[r, c])
            tp += 1

    fp = len(pred_ids) - tp
    fn = len(gt_ids)  - tp

    sq = np.mean(matched_iou) if matched_iou else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-10)
    pq = sq * rq

    # per-point coverage: fraction of points whose GT label was matched correctly
    matched_gt   = [gt_ids[r]   for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold]
    matched_pred = [pred_ids[c] for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold]

    correct = np.zeros(len(gt_labels), dtype=bool)
    for g, p in zip(matched_gt, matched_pred):
        mask = gt_labels == g
        correct[mask] = pred_labels[mask] == p
    coverage = correct.mean()

    return {
        "PQ":       round(pq,       4),
        "SQ":       round(sq,       4),
        "RQ":       round(rq,       4),
        "TP":       tp,
        "FP":       fp,
        "FN":       fn,
        "coverage": round(coverage, 4),
        "n_gt":     len(gt_ids),
        "n_pred":   len(pred_ids),
    }