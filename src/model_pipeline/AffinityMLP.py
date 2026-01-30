import torch
import torch.nn as nn
from typing import Dict, Any


class AffinityMLP(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
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
    
    @classmethod
    def from_config(cls, config_path: str) -> 'AffinityMLP':
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(cfg.get('affinity_mlp', {}))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
