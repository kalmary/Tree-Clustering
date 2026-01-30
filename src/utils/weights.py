import sys
import torch
from typing import Optional
from collections import Counter


def calculate_binary_weights(
    loader: torch.utils.data.DataLoader,
    total: Optional[int] = None,
    verbose: bool = True,
    return_pos_weight: bool = True
) -> float:
    """
    Calculates weight for binary classification.
    
    Args:
        loader: DataLoader yielding (features, labels) where labels are 0/1
        total: Total iterations (for progress display)
        verbose: Print statistics
        return_pos_weight: If True, returns pos_weight for BCEWithLogitsLoss (count_neg/count_pos)
                          If False, returns alpha for Focal loss (count_neg/total)
        
    Returns:
        weight: pos_weight (ratio) or alpha (proportion) depending on return_pos_weight
    """
    count_neg = 0
    count_pos = 0

    if verbose:
        print("\nCalculating binary class weights...")
    
    for i, (_, targets) in enumerate(loader):
        count_neg += (targets == 0).sum().item()
        count_pos += (targets == 1).sum().item()
        
        del targets

        if verbose and i % 10 == 0:
            sys.stdout.write(f"\rProcessing iteration: {i}/{total if total else '?'}")
            sys.stdout.flush()
    
    if verbose:
        sys.stdout.write(f"\n")
    
    total_samples = count_neg + count_pos
    
    if verbose:
        print(f"\nClass distribution:")
        print(f"  Class 0 (negative): {int(count_neg):8d} samples ({count_neg/total_samples*100:5.2f}%)")
        print(f"  Class 1 (positive): {int(count_pos):8d} samples ({count_pos/total_samples*100:5.2f}%)")
    
    if count_pos == 0:
        if verbose:
            print("Warning: No positive samples found!")
        return 1.0
    
    if return_pos_weight:
        # For BCEWithLogitsLoss pos_weight parameter
        weight = count_neg / count_pos
        if verbose:
            print(f"\npos_weight: {weight:.4f}")
            print(f"  (use with BCEWithLogitsLoss(pos_weight=...))")
    else:
        # For Focal loss alpha parameter
        weight = count_neg / total_samples
        if verbose:
            print(f"\nAlpha weight: {weight:.4f}")
            print(f"  (use with FocalLoss(alpha=...))")
    
    return weight
