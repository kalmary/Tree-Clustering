from scipy.optimize import linear_sum_assignment
import numpy as np

def evaluate_segmentation(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """
    Permutation-invariant metrics robust to over-segmentation.
    
    - GT coverage:   for each GT instance, fraction of its points claimed by
                     its best-matching predicted cluster (mean over GT instances)
    - Pred coverage: for each predicted cluster, fraction of its points that
                     belong to its best-matching GT instance (mean over pred clusters)
    - F1 coverage:   harmonic mean of the two above
    - Over-seg ratio: n_pred / n_gt  (>1 means over-segmented)
    - Mean best IoU: mean over GT instances of their best IoU with any pred cluster
    """
    gt_ids   = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)

    # --- GT coverage: how well each GT tree is covered by its best pred match ---
    gt_coverages  = []
    best_iou_list = []

    for g in gt_ids:
        gt_mask  = gt_labels == g
        gt_size  = gt_mask.sum()
        best_overlap = 0
        best_iou     = 0.0

        for p in pred_ids:
            pred_mask    = pred_labels == p
            intersection = (gt_mask & pred_mask).sum()
            if intersection == 0:
                continue
            best_overlap = max(best_overlap, intersection)
            union        = (gt_mask | pred_mask).sum()
            best_iou     = max(best_iou, intersection / union)

        gt_coverages.append(best_overlap / gt_size)
        best_iou_list.append(best_iou)

    # --- pred coverage: how pure each predicted cluster is ---
    pred_coverages = []
    for p in pred_ids:
        pred_mask  = pred_labels == p
        pred_size  = pred_mask.sum()
        best_overlap = 0

        for g in gt_ids:
            gt_mask      = gt_labels == g
            intersection = (gt_mask & pred_mask).sum()
            best_overlap = max(best_overlap, intersection)

        pred_coverages.append(best_overlap / pred_size)

    gt_cov   = float(np.mean(gt_coverages))
    pred_cov = float(np.mean(pred_coverages))
    f1_cov   = 2 * gt_cov * pred_cov / (gt_cov + pred_cov + 1e-10)

    return {
        "gt_coverage":    round(gt_cov,                  4),  # recall-like:  are GT trees fully captured?
        "pred_coverage":  round(pred_cov,                4),  # precision-like: are pred clusters pure?
        "f1_coverage":    round(f1_cov,                  4),  # balance of both
        "mean_best_iou":  round(float(np.mean(best_iou_list)), 4),
        "over_seg_ratio": round(len(pred_ids) / len(gt_ids),   4),  # 1.0 = perfect, >1 = over-seg
        "n_gt":           len(gt_ids),
        "n_pred":         len(pred_ids),
    }