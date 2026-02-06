import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class AffinityMLP(nn.Module):
    def __init__(self, config: Dict[str, Any], scaling_config: Optional[Dict[str, Any]] = None, scaling_method: str = "standard") -> None:
        super().__init__()
        
        input_dim = config.get('input_dim', 4)
        hidden_dims = config.get('hidden_dims', [32, 16])
        dropout = config.get('dropout', 0.0)
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        self.apply_scaling = False
        self.scaling_method = scaling_method
        if scaling_config is not None:
            self._setup_scaling(scaling_config, scaling_method)

    def _setup_scaling(self, scaling_params: Optional[Dict[str, Any]], method: str):
        """Setup scaling parameters as buffers (non-trainable tensors)."""
        self.apply_scaling = True
        if scaling_params is None or method == 'none':
            self.register_buffer('scale_mean', None)
            self.register_buffer('scale_std', None)
            self.register_buffer('scale_min', None)
            self.register_buffer('scale_max', None)
            self.register_buffer('scale_median', None)
            self.register_buffer('scale_iqr', None)
            return
        
        if method == 'standard':
            means = torch.tensor(
                scaling_params['standard_scaling']['means'], 
                dtype=torch.float32
            )
            stds = torch.tensor(
                scaling_params['standard_scaling']['stds'], 
                dtype=torch.float32
            )
            self.register_buffer('scale_mean', means)
            self.register_buffer('scale_std', stds)
            self.register_buffer('scale_min', None)
            self.register_buffer('scale_max', None)
            self.register_buffer('scale_median', None)
            self.register_buffer('scale_iqr', None)
        
        elif method == 'minmax':
            mins = torch.tensor(
                scaling_params['minmax_scaling']['mins'], 
                dtype=torch.float32
            )
            maxs = torch.tensor(
                scaling_params['minmax_scaling']['maxs'], 
                dtype=torch.float32
            )
            self.register_buffer('scale_mean', None)
            self.register_buffer('scale_std', None)
            self.register_buffer('scale_min', mins)
            self.register_buffer('scale_max', maxs)
            self.register_buffer('scale_median', None)
            self.register_buffer('scale_iqr', None)
        
        elif method == 'robust':
            medians = torch.tensor(
                scaling_params['robust_scaling']['medians'], 
                dtype=torch.float32
            )
            q25s = torch.tensor(
                scaling_params['robust_scaling']['q25s'], 
                dtype=torch.float32
            )
            q75s = torch.tensor(
                scaling_params['robust_scaling']['q75s'], 
                dtype=torch.float32
            )
            iqrs = q75s - q25s
            self.register_buffer('scale_mean', None)
            self.register_buffer('scale_std', None)
            self.register_buffer('scale_min', None)
            self.register_buffer('scale_max', None)
            self.register_buffer('scale_median', medians)
            self.register_buffer('scale_iqr', iqrs)
        
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def scale_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale features using the configured method.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
        
        Returns:
            Scaled tensor of same shape
        """
        if self.scaling_method == 'none':
            return x
        
        elif self.scaling_method == 'standard':
            # Z-score normalization: (x - mean) / std
            return (x - self.scale_mean) / (self.scale_std + 1e-8)
        
        elif self.scaling_method == 'minmax':
            # Min-max scaling: (x - min) / (max - min)
            return (x - self.scale_min) / (self.scale_max - self.scale_min + 1e-8)
        
        elif self.scaling_method == 'robust':
            # Robust scaling: (x - median) / IQR
            return (x - self.scale_median) / (self.scale_iqr + 1e-8)
        
        return x
    
    @classmethod
    def from_config(cls, config_path: str) -> 'AffinityMLP':
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(cfg.get('affinity_mlp', {}))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.apply_scaling:
            x = self.scale_features(x)

        return self.net(x).squeeze(-1)


def test_affinity_mlp():
    device = torch.device("cpu")
    config = {'input_dim': 4, 'hidden_dims': [32, 16], 'dropout': 0.0}
    model = AffinityMLP(config).to(device)
    
    # Dummy edge features: [distance, angle, thickness_ratio, vertical_diff]
    dummy_input = torch.tensor([
        [0.5, 0.9, 0.8, 0.1],
        [1.2, 0.7, 0.6, 0.3],
        [0.3, 0.95, 0.9, 0.05]
    ], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    output = output.cpu().numpy()
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"


if __name__ == "__main__":
    test_affinity_mlp()