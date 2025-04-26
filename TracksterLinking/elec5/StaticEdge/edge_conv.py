import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph



class CustomStaticEdgeConv(nn.Module):
    def __init__(self, nn_module):
        super(CustomStaticEdgeConv, self).__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F).
            edge_index (torch.Tensor): Predefined edges [2, E], where E is the number of edges.
        
        Returns:
            torch.Tensor: Node features after static edge aggregation.
        """
        row, col = edge_index  # Extract row (source) and col (target) nodes
        x_center = x[row]
        x_neighbor = x[col]
        
        # Compute edge features (relative)
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)
        edge_features = self.nn_module(edge_features)

        # Aggregate features back to nodes
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        # Normalization (Divide by node degrees)
        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        node_features = node_features / counts

        return node_features


