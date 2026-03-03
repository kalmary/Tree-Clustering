import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        margin: float = 0.1,
    ):
        """
        Contrastive loss on node embeddings using edge supervision.

        Pulls same-tree node embedding pairs toward cosine similarity 1.0.
        Pushes inter-tree node embedding pairs below a margin.

        Weighted by alpha using the same convention as focal loss:
            alpha         = weight for class-1 (same-tree)
            (1 - alpha)   = weight for class-0 (inter-tree)

        For majority class-1 (your case, 6.4:1):
            alpha < 0.5 — downweights same-tree term
            alpha_exact = n_c0 / (n_c0 + n_c1) = 0.135

        Args:
            alpha:  Class-1 weight. Use focal_params() to compute exact value.
            margin: Inter-tree pairs are penalised only if cosine similarity
                    exceeds this margin. 0.1 means pairs are allowed to be
                    slightly similar before penalty kicks in.
        """
        super().__init__()
        
        self.alpha  = alpha
        self.margin = margin

    def forward(
        self,
        node_embeddings: torch.Tensor,   # (N, hidden_dim)
        edge_index:      torch.Tensor,   # (2, E)
        edge_labels:     torch.Tensor,   # (E,) — 0 or 1
    ) -> torch.Tensor:
        h_u = node_embeddings[edge_index[0]]   # (E, hidden_dim)
        h_v = node_embeddings[edge_index[1]]   # (E, hidden_dim)

        cos_sim = F.cosine_similarity(h_u, h_v, dim=1)   # (E,)

        same_tree  = edge_labels == 1
        inter_tree = edge_labels == 0

        # same-tree: minimise (1 - cos_sim) → push similarity toward 1
        loss_same = (
            (1.0 - cos_sim[same_tree]).mean()
            if same_tree.any() else cos_sim.new_tensor(0.0)
        )

        # inter-tree: penalise similarity above margin → push similarity below margin
        loss_inter = (
            F.relu(cos_sim[inter_tree] - self.margin).mean()
            if inter_tree.any() else cos_sim.new_tensor(0.0)
        )

        return self.alpha * loss_same + (1.0 - self.alpha) * loss_inter