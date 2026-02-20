import numpy as np

def evaluate_segmentation(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    gt_ids   = np.unique(gt_labels)
    pred_ids = np.unique(pred_labels)

    gt_sizes  = {g: int((gt_labels == g).sum())   for g in gt_ids}
    pred_sizes = {p: int((pred_labels == p).sum()) for p in pred_ids}
    total_gt_points   = sum(gt_sizes.values())
    total_pred_points = sum(pred_sizes.values())

    # --- GT coverage (recall-like) ---
    gt_coverages  = []
    best_iou_list = []
    weighted_iou  = []

    for g in gt_ids:
        gt_mask      = gt_labels == g
        gt_size      = gt_sizes[g]
        best_overlap = 0
        best_iou     = 0.0

        for p in pred_ids:
            pred_mask    = pred_labels == p
            intersection = int((gt_mask & pred_mask).sum())
            if intersection == 0:
                continue
            best_overlap = max(best_overlap, intersection)
            union        = int((gt_mask | pred_mask).sum())
            best_iou     = max(best_iou, intersection / union)

        gt_coverages.append(best_overlap / gt_size)
        best_iou_list.append(best_iou)
        weighted_iou.append(best_iou * gt_size / total_gt_points)

    # --- pred purity weighted by cluster size ---
    pred_purities = []
    pred_weights  = []

    for p in pred_ids:
        pred_mask    = pred_labels == p
        pred_size    = pred_sizes[p]
        best_overlap = 0

        for g in gt_ids:
            intersection = int(((gt_labels == g) & pred_mask).sum())  # explicit parentheses
            best_overlap = max(best_overlap, intersection)

        pred_purities.append(best_overlap / pred_size)
        pred_weights.append(pred_size / total_pred_points)

    gt_cov   = float(np.mean(gt_coverages))
    pred_pur = float(np.average(pred_purities, weights=pred_weights))
    f1_cov   = 2 * gt_cov * pred_pur / (gt_cov + pred_pur + 1e-10)

    # --- seg_quality: f1 penalised by deviation from ratio=1 ---
    ratio         = len(pred_ids) / (len(gt_ids) + 1e-10)
    ratio_penalty = 1.0 / (1.0 + abs(np.log(ratio)))
    seg_quality   = f1_cov * ratio_penalty

    return {
        "gt_coverage":       round(gt_cov,                        4),
        "pred_purity":       round(pred_pur,                      4),
        "f1_coverage":       round(f1_cov,                        4),
        "seg_quality":       round(seg_quality,                   4),
        "mean_best_iou":     round(float(np.mean(best_iou_list)), 4),
        "weighted_best_iou": round(float(np.sum(weighted_iou)),   4),
        "over_seg_ratio":    round(ratio,                         4),
        "n_gt":              len(gt_ids),
        "n_pred":            len(pred_ids),
    }