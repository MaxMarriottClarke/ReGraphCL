import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph



class CustomDynamicEdgeConv(nn.Module):
    def __init__(self, nn_module, k):
        super(CustomDynamicEdgeConv, self).__init__()
        self.nn_module = nn_module  
        self.k = k  

    def forward(self, x, batch):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F), where N is the number of nodes.
            batch (torch.Tensor): Batch indices mapping each node to a graph.

        Returns:
            torch.Tensor: Node features after aggregation.
        """
        
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)  

        
        row, col = edge_index  

        
        x_center = x[row]  
        x_neighbor = x[col]  
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)  

        
        edge_features = self.nn_module(edge_features)  

        
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)  

        
        counts = torch.bincount(row, minlength=num_nodes).view(-1, 1).clamp(min=1)
        node_features = node_features / counts  

        return node_features
