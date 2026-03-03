import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class EdgeClassifierGNN(nn.Module):
    def __init__(self, config, scaling_params=None):
        """
        Binary edge classifier using GNN.
        
        Args:
            config: dict with model hyperparameters
            scaling_params: dict with feature scaling parameters (optional).
                            Must contain node_<method>_scaling and edge_<method>_scaling keys.
        """
        super(EdgeClassifierGNN, self).__init__()
        
        self.node_feat_dim = config['node_feat_dim']
        self.edge_feat_dim = config['edge_feat_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.gat_heads = config['gat_heads']
        self.scaling_method = config['scaling_method'] if scaling_params is not None else None
        
        if scaling_params is not None:
            self._register_scaling_buffers(scaling_params)
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.convs.append(
                GATConv(
                    self.hidden_dim,
                    self.hidden_dim // self.gat_heads,
                    heads=self.gat_heads,
                    edge_dim=self.edge_feat_dim
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Edge classifier
        edge_input_dim = self.hidden_dim * 2 + self.edge_feat_dim
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
    
    def _register_scaling_buffers(self, scaling_params):
        """
        Register scaling parameters as non-trainable buffers.
        Expects keys like 'node_robust_scaling', 'edge_robust_scaling', etc.
        Node params must have length == node_feat_dim (10).
        Edge params must have length == edge_feat_dim (12).
        """
        method = self.scaling_method
        node_key = f'node_{method}_scaling'
        edge_key = f'edge_{method}_scaling'

        if node_key not in scaling_params:
            raise KeyError(
                f"Expected key '{node_key}' in scaling_params, but only found: "
                f"{list(scaling_params.keys())}. "
                f"Re-run analyze_graph_data.py to regenerate the JSON with separate node/edge keys."
            )
        if edge_key not in scaling_params:
            raise KeyError(
                f"Expected key '{edge_key}' in scaling_params, but only found: "
                f"{list(scaling_params.keys())}."
            )

        node_p = scaling_params[node_key]
        edge_p = scaling_params[edge_key]

        if method == 'standard':
            self.register_buffer('node_means', torch.tensor(node_p['means'], dtype=torch.float32))
            self.register_buffer('node_stds',  torch.tensor(node_p['stds'],  dtype=torch.float32))
            self.register_buffer('edge_means', torch.tensor(edge_p['means'], dtype=torch.float32))
            self.register_buffer('edge_stds',  torch.tensor(edge_p['stds'],  dtype=torch.float32))

        elif method == 'minmax':
            self.register_buffer('node_mins', torch.tensor(node_p['mins'], dtype=torch.float32))
            self.register_buffer('node_maxs', torch.tensor(node_p['maxs'], dtype=torch.float32))
            self.register_buffer('edge_mins', torch.tensor(edge_p['mins'], dtype=torch.float32))
            self.register_buffer('edge_maxs', torch.tensor(edge_p['maxs'], dtype=torch.float32))

        elif method == 'robust':
            self.register_buffer('node_medians', torch.tensor(node_p['medians'], dtype=torch.float32))
            self.register_buffer('node_q25s',    torch.tensor(node_p['q25s'],    dtype=torch.float32))
            self.register_buffer('node_q75s',    torch.tensor(node_p['q75s'],    dtype=torch.float32))
            self.register_buffer('edge_medians', torch.tensor(edge_p['medians'], dtype=torch.float32))
            self.register_buffer('edge_q25s',    torch.tensor(edge_p['q25s'],    dtype=torch.float32))
            self.register_buffer('edge_q75s',    torch.tensor(edge_p['q75s'],    dtype=torch.float32))

        # Sanity check shapes at init time
        self._verify_buffer_shapes()

    def _verify_buffer_shapes(self):
        """Raise clearly if buffer dims don't match config dims."""
        method = self.scaling_method
        if method == 'standard':
            assert self.node_means.shape[0] == self.node_feat_dim, (
                f"node_means has {self.node_means.shape[0]} values but node_feat_dim={self.node_feat_dim}")
            assert self.edge_means.shape[0] == self.edge_feat_dim, (
                f"edge_means has {self.edge_means.shape[0]} values but edge_feat_dim={self.edge_feat_dim}")
        elif method == 'minmax':
            assert self.node_mins.shape[0] == self.node_feat_dim
            assert self.edge_mins.shape[0] == self.edge_feat_dim
        elif method == 'robust':
            assert self.node_medians.shape[0] == self.node_feat_dim, (
                f"node_medians has {self.node_medians.shape[0]} values but node_feat_dim={self.node_feat_dim}")
            assert self.edge_medians.shape[0] == self.edge_feat_dim, (
                f"edge_medians has {self.edge_medians.shape[0]} values but edge_feat_dim={self.edge_feat_dim}")

    def scale_node_features(self, x):
        """Scale node features [N, node_feat_dim] using the configured method."""
        if self.scaling_method is None:
            return x
        if self.scaling_method == 'standard':
            return (x - self.node_means) / (self.node_stds + 1e-8)
        elif self.scaling_method == 'minmax':
            return (x - self.node_mins) / (self.node_maxs - self.node_mins + 1e-8)
        elif self.scaling_method == 'robust':
            iqr = self.node_q75s - self.node_q25s
            return (x - self.node_medians) / (iqr + 1e-8)
        return x

    def scale_edge_features(self, edge_attr):
        """Scale edge features [E, edge_feat_dim] using the configured method."""
        if self.scaling_method is None:
            return edge_attr
        if self.scaling_method == 'standard':
            return (edge_attr - self.edge_means) / (self.edge_stds + 1e-8)
        elif self.scaling_method == 'minmax':
            return (edge_attr - self.edge_mins) / (self.edge_maxs - self.edge_mins + 1e-8)
        elif self.scaling_method == 'robust':
            iqr = self.edge_q75s - self.edge_q25s
            return (edge_attr - self.edge_medians) / (iqr + 1e-8)
        return edge_attr

    def forward(self, data, return_embeddings: bool = False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x         = self.scale_node_features(x)
        edge_attr = self.scale_edge_features(edge_attr)
        x         = self.node_encoder(x)

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr=edge_attr)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new if i > 0 else x_new

        src_embeddings = x[edge_index[0]]
        dst_embeddings = x[edge_index[1]]
        edge_input     = torch.cat([src_embeddings, dst_embeddings, edge_attr], dim=1)
        logits         = self.edge_classifier(edge_input).squeeze(-1)

        if return_embeddings:
            return logits, x, edge_index   # x used for contrastive loss
        return logits


if __name__ == "__main__":
    config = {
        'node_feat_dim': 11,
        'edge_feat_dim': 9,
        'hidden_dim': 128,
        'num_layers': 4,
        'dropout': 0.2,
        'gat_heads': 4,
        'scaling_method': 'robust'
    }
    
    model = EdgeClassifierGNN(config, scaling_params=None)
    model.eval()
    
    data = Data(
        x=torch.randn(20, 11),
        edge_index=torch.randint(0, 20, (2, 100)),
        edge_attr=torch.randn(100, 9)
    )
    with torch.no_grad():
        logits = model(data)
    print(f"Output shape: {logits.shape}")
    print(f"Sample predictions: {torch.sigmoid(logits[:5])}")