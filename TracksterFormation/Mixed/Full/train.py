import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import math

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import knn_graph

import torch
import torch.nn as nn
import math

class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.3):
        """
        A graph transformer layer that uses provided edge indices to restrict attention.
        This layer computes multi-head attention only over edges given in edge_index.

        Args:
            hidden_dim (int): Dimension of node features.
            num_heads (int): Number of attention heads. Must evenly divide hidden_dim.
            dropout (float): Dropout rate for attention weights and residuals.
        """
        super(GraphTransformerLayer, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        # Linear projections for queries, keys, and values.
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(dropout)

        # Layer normalization and feed-forward network.
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the graph transformer layer using edge indices.

        Args:
            x (torch.Tensor): Node features of shape (N, hidden_dim).
            edge_index (torch.Tensor): Edge indices of shape (2, E), where each column
                                       represents an edge from source to target.

        Returns:
            torch.Tensor: Updated node features of shape (N, hidden_dim).
        """
        N = x.size(0)
        src, tgt = edge_index  # src: source node indices, tgt: target node indices

        # Compute queries, keys, and values and reshape for multi-head attention.
        # Shape: (N, num_heads, d_k)
        Q = self.q_linear(x).view(N, self.num_heads, self.d_k)
        K = self.k_linear(x).view(N, self.num_heads, self.d_k)
        V = self.v_linear(x).view(N, self.num_heads, self.d_k)

        # For each edge, use the target node's query and the source node's key.
        # q: (E, num_heads, d_k), k: (E, num_heads, d_k)
        q = Q[tgt]
        k = K[src]

        # Compute scaled dot-product attention scores for each edge.
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d_k)  # (E, num_heads)

        # For numerical stability and per-target softmax, we need the maximum score per target node.
        # Create a tensor for max scores, then use scatter_reduce.
        scores_max = torch.full((N, self.num_heads), -float('inf'), device=x.device)
        scores_max = scores_max.scatter_reduce(0, 
                                               tgt.unsqueeze(-1).expand(-1, self.num_heads),
                                               scores, 
                                               reduce="amax", 
                                               include_self=True)
        # Subtract max from each score (indexed by tgt) and exponentiate.
        scores = scores - scores_max[tgt]
        scores_exp = torch.exp(scores)

        # Compute the sum of exponentials per target node.
        scores_sum = torch.zeros(N, self.num_heads, device=x.device)
        scores_sum = scores_sum.scatter_add(0, tgt.unsqueeze(-1).expand(-1, self.num_heads), scores_exp)

        # Normalize scores to get attention coefficients.
        alpha = scores_exp / (scores_sum[tgt] + 1e-16)  # (E, num_heads)
        alpha = self.att_dropout(alpha)

        # Multiply attention coefficients with values from the source nodes.
        messages = alpha.unsqueeze(-1) * V[src]  # (E, num_heads, d_k)

        # Aggregate messages for each target node.
        out = torch.zeros(N, self.num_heads, self.d_k, device=x.device)
        out = out.scatter_add(0, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.d_k), messages)

        # Concatenate heads and project.
        out = out.view(N, self.hidden_dim)
        out = self.out_linear(out)

        # Residual connection and layer normalization.
        x = self.layer_norm1(x + self.dropout(out))

        # Feed-forward network with residual connection.
        ff_out = self.ff(x)
        x = self.layer_norm2(x + self.dropout(ff_out))
        return x

# Example network using the above GraphTransformerLayer
class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, num_heads=4):
        """
        Initializes a graph transformer network that uses edge indices for attention.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Output dimension of the final layer.
            num_heads (int): Number of attention heads.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input encoder (assumes input features of size 16).
        self.lc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Stack of graph transformer layers.
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Output layer.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the graph transformer network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 16).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor, optional): Batch vector for nodes.

        Returns:
            tuple: (Output features, Batch vector)
        """
        # Encode input features.
        x_enc = self.lc_encode(x)  # (N, hidden_dim)

        # Apply each transformer layer using the provided edge_index.
        for layer in self.transformer_layers:
            x_enc = layer(x_enc, edge_index)

        # Final output transformation.
        out = self.output(x_enc)
        return out, batch


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import knn_graph
from torch_geometric.data import DataLoader

def contrastive_loss_curriculum_both(embeddings, group_ids, temperature=0.1, alpha=0.0):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining, using group_ids only.

    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[j]),
          where j is a randomly chosen index (≠ i) such that group_ids[j] == group_ids[i].
      - Hard positive similarity: [omitted in this simplified version]
      - Blended positive similarity: [omitted in this simplified version; we just use pos_sim_orig]

      - Random negative similarity: rand_neg_sim = sim(embeddings[i], embeddings[k]),
          where k is a randomly chosen index such that group_ids[k] != group_ids[i].
      - Hard negative similarity: max { sim(embeddings[i], embeddings[k]) : 
                                       group_ids[k] != group_ids[i] }
      - Blended negative similarity: (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    The loss per anchor is:
         loss_i = - log( exp(pos_sim_orig / temperature) / 
                         ( exp(pos_sim_orig / temperature) + exp(blended_neg / temperature) ) )

    Anchors that lack any valid positives or negatives contribute 0.

    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter between random and hard negative mining.

    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings so cosine similarity is just the dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()  # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    
    # --- Positives ---
    # Build a mask for positives: same group (excluding self)
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask.fill_diagonal_(False)  # exclude self from positives
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)

    # Select a random positive candidate in a vectorized way:
    #   1) Generate uniform random values in [0,1].
    #   2) Multiply by pos_mask to zero out invalid positions.
    #   3) Subtract (1 - pos_mask) so those invalid positions become negative.
    #   4) argmax finds the index of the largest random among valid spots.
    rand_vals_pos = torch.rand_like(sim_matrix)
    rand_vals_pos = rand_vals_pos * pos_mask.float() - (1 - pos_mask.float())
    rand_pos_indices = torch.argmax(rand_vals_pos, dim=1)

    # Gather the chosen random positives. Fallback to self if no valid positives.
    rand_pos_sim = sim_matrix[idx, rand_pos_indices]
    pos_sim_orig = torch.where(no_valid_pos, sim_matrix[idx, idx], rand_pos_sim)

    # We skip the hard positive logic here. So final is just:
    blended_pos = pos_sim_orig

    # --- Negatives ---
    # Build a mask for negatives: indices where group_ids differ
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    valid_neg_counts = neg_mask.sum(dim=1)
    no_valid_neg = (valid_neg_counts == 0)

    # Random negative similarity: choose a random negative candidate per anchor
    rand_vals_neg = torch.rand_like(sim_matrix)
    rand_vals_neg = rand_vals_neg * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals_neg, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]

    # Hard negative similarity: maximum similarity among negatives
    sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, -float('inf'))
    hard_neg_sim, _ = sim_matrix_neg.max(dim=1)
    # If no valid negatives, store sentinel of -1.0
    hard_neg_sim = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)

    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    # --- Loss Computation ---
    numerator = torch.exp(blended_pos / temperature)
    denominator = numerator + torch.exp(blended_neg / temperature)
    loss = -torch.log(numerator / denominator)

    # Anchors with no valid negatives get zero loss
    loss = loss.masked_fill(no_valid_neg, 0.0)

    return loss.mean()




def contrastive_loss_curriculum(embeddings, group_ids, temperature=0.5, alpha=0.0):
    """
    Curriculum loss that uses both positive and negative blending based solely on group_ids.
    
    Args:
        embeddings: Tensor of shape (N, D).
        group_ids: 1D Tensor (length N).
        temperature: Temperature scaling factor.
        alpha: Blending parameter.
        
    Returns:
        Scalar loss.
    """
    return contrastive_loss_curriculum_both(embeddings, group_ids, temperature, alpha)



#################################
# Training and Testing Functions
#################################


def train_new(train_loader, model, optimizer, k_value, device, alpha):
    model.train()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Convert data.assoc to tensor if needed.
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device)
                                          for a in data.assoc])
            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc

        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        #edge_index = build_knn_edge_index(data.x[:, :3], data.x_batch, k_value)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        # Partition batch by event.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_group_ids, temperature=0.1, alpha = alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, k_value, device, alpha):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device)
                                          for a in data.assoc])
            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        #edge_index = build_knn_edge_index(data.x[:, :3], data.x_batch, k_value)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_group_ids, temperature=0.1, alpha = alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)