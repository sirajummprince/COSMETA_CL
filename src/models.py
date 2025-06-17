import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, SAGEConv, SGConv, 
    global_mean_pool, global_max_pool, global_add_pool, 
    SAGPooling, GraphNorm, BatchNorm, LayerNorm
)

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, 
                 encoder_type='GCN', pool_type='mean_pool', 
                 num_layers=3, dropout=0.5, use_norm='batch'):
        super(GNNEncoder, self).__init__()
        self.encoder_type = encoder_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels
            
            if encoder_type == 'GAT':
                self.convs.append(GATConv(in_dim, out_dim, heads=4, concat=False))
            elif encoder_type == 'GraphSAGE':
                self.convs.append(SAGEConv(in_dim, out_dim))
            elif encoder_type == 'SGC':
                self.convs.append(SGConv(in_dim, out_dim, K=2))
            elif encoder_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
                self.convs.append(GINConv(mlp))
            else:  # GCN (default)
                self.convs.append(GCNConv(in_dim, out_dim))
            
            # Add normalization
            if use_norm == 'batch':
                self.norms.append(BatchNorm(out_dim))
            elif use_norm == 'layer':
                self.norms.append(LayerNorm(out_dim))
            elif use_norm == 'graph':
                self.norms.append(GraphNorm(out_dim))
            else:
                self.norms.append(nn.Identity())
        
        # Output layers
        self.lins = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, output_channels)
        )
        
        # Graph-level pooling
        self.pool = Pool(hidden_channels, type=pool_type)
        
        # Residual connections
        self.residual = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        for layer in self.lins:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.residual is not None:
            self.residual.reset_parameters()
        if hasattr(self.pool, 'reset_parameters'):
            self.pool.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        # Store original features for residual
        x_orig = x
        
        # Apply GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection after first layer
            if i == 0 and self.residual is not None:
                x_orig = self.residual(x_orig)
                x = x + x_orig
        
        # Node-level output
        out = self.lins(x)
        
        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.pool(x, edge_index, batch)
        
        return out, pooled

    def get_params(self):
        return OrderedDict((name, param) for name, param in self.named_parameters())


class Pool(nn.Module):
    def __init__(self, in_channels, type='mean_pool', ratio=0.8):
        super(Pool, self).__init__()
        self.type = type
        if type == 'sag_pool':
            self.sag_pool = SAGPooling(in_channels, ratio)
        elif type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(in_channels, 1),
                nn.Tanh()
            )
        
    def reset_parameters(self):
        if hasattr(self, 'sag_pool'):
            self.sag_pool.reset_parameters()
        if hasattr(self, 'attention'):
            for layer in self.attention:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
    
    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        if self.type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif self.type == 'max_pool':
            return global_max_pool(x, batch)
        elif self.type == 'sum_pool':
            return global_add_pool(x, batch)
        elif self.type == 'sag_pool':
            x, _, _, batch, _, _ = self.sag_pool(x, edge_index, batch=batch)
            return global_mean_pool(x, batch)
        elif self.type == 'attention':
            weights = self.attention(x)
            weighted_x = x * weights
            return global_mean_pool(weighted_x, batch)
        else:  # mean-max concatenation
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=-1)


class ProtoNet(nn.Module):
    """Prototypical Network for few-shot learning"""
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        
    def forward(self, support_x, support_y, query_x, support_edge_index, query_edge_index):
        # Encode support and query
        support_features, _ = self.encoder(support_x, support_edge_index)
        query_features, _ = self.encoder(query_x, query_edge_index)
        
        # Compute prototypes
        unique_classes = torch.unique(support_y)
        prototypes = []
        
        for c in unique_classes:
            mask = support_y == c
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances
        distances = torch.cdist(query_features, prototypes)
        
        return -distances  # Negative distance as logits