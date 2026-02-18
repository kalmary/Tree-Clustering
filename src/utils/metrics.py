import torch
import torch.nn as nn


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


class FocalLossBCE(nn.Module):
    def __init__(self, pos_weight=1.0, gamma=2.0):
        super().__init__()
        self.gamma = gamma

        if isinstance(pos_weight, torch.Tensor):
            self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_weight))

    def forward(self, inputs, targets):
        assert not inputs.isnan().any(), "NaN in inputs"
        assert not inputs.isinf().any(), "Inf in inputs"
        assert targets.min() >= 0 and targets.max() <= 1, "Bad target values"

        inputs = inputs.float()
        inputs = inputs.float()

        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
