import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, SAGEConv

class EdgeGNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Node feature encoder
        self.conv1 = GCNConv(n_features, 64)
        self.conv2 = GCNConv(64, 32)
        
        # Edge classifier
        self.edge_mlp = nn.Sequential(
            nn.Linear(64, 32),  # 32 + 32 from both nodes
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x, edge_index, edge_features):
        # Encode nodes
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # For each edge, concatenate source and target node embeddings
        row, col = edge_index
        edge_repr = torch.cat([x[row], x[col]], dim=-1)
        
        # Classify edge
        return self.edge_mlp(edge_repr)