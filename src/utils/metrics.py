import torch


def binary_f1_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute F1 score for binary classification.
    
    Args:
        preds: Predictions (logits or probabilities), shape (N,)
        targets: Ground truth labels (0 or 1), shape (N,)
        threshold: Classification threshold (default 0.5)
        
    Returns:
        f1: F1 score (float)
    """
    # Binarize predictions
    preds_binary = (preds > threshold).float()
    
    # Compute TP, FP, FN
    tp = ((preds_binary == 1) & (targets == 1)).sum().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().item()
    
    # Compute precision and recall
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # Compute F1
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return f1
