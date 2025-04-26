# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define linear layers for query, key, and value
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        """
        x: Tensor of shape (N, embed_dim)
        mask: Tensor of shape (N, N) where mask[i, j] = 1 if node j can be attended by node i
        """
        N, D = x.size()
        
        Q = self.W_Q(x)  # (N, D)
        K = self.W_K(x)  # (N, D)
        V = self.W_V(x)  # (N, D)
        
        # Reshape for multi-head: (num_heads, N, head_dim)
        Q = Q.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N, head_dim)
        K = K.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N, head_dim)
        V = V.view(N, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, N, head_dim)
        
        # Compute scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)  # (num_heads, N, N)
        
        if mask is not None:
            # mask shape: (N, N)
            # Expand mask to (num_heads, N, N)
            mask = mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (num_heads, N, N)
        
        # Apply attention weights to V
        attn_output = torch.bmm(attn_weights, V)  # (num_heads, N, head_dim)
        
        # Concatenate all heads
        attn_output = attn_output.transpose(0, 1).contiguous().view(N, D)  # (N, D)
        
        # Final linear projection
        output = self.out_proj(attn_output)  # (N, D)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block
        attn_output = self.attention(x, mask)  # (N, D)
        x = self.layer_norm1(x + self.dropout(attn_output))  # (N, D)
        
        # Feed-forward block
        ffn_output = self.ffn(x)  # (N, D)
        x = self.layer_norm2(x + self.dropout(ffn_output))  # (N, D)
        
        return x

class GraphTransformerNet(nn.Module):
    def __init__(self, input_dim=15, embed_dim=64, num_heads=4, ff_hidden_dim=128, 
                 num_layers=3, dropout=0.1, contrastive_dim=8):
        super(GraphTransformerNet, self).__init__()
        self.embed_dim = embed_dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU()
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output projection for contrastive learning
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, contrastive_dim)
        )
        
    def forward(self, x, mask=None):
        """
        x: Tensor of shape (N, input_dim) for a single graph/event
        mask: Tensor of shape (N, N), optional for masking attention
        """
        # Encode node features
        x = self.node_encoder(x)  # (N, embed_dim)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)  # (N, embed_dim)
        
        # Project to contrastive dimension
        out = self.output_proj(x)  # (N, contrastive_dim)
        
        return out

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, contrastive_dim=8, num_heads=4, ff_hidden_dim=128):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        
        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        # Graph Transformer
        self.graph_transformer = GraphTransformerNet(
            input_dim=hidden_dim,  # Since lc_encode outputs hidden_dim
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            contrastive_dim=contrastive_dim
        )
        
    def forward(self, x, batch=None):
        """
        x: Tensor of shape (N, 15) for a single event
        edge_index: Not used since we're using fully connected graphs
        batch: Not used in this simplified version
        """
        # Encode input features
        x_lc_enc = self.lc_encode(x)  # (N, hidden_dim)
        
        # Since we're using fully connected graphs, create a mask with all ones
        N = x_lc_enc.size(0)
        mask = torch.ones((N, N), device=x.device)  # (N, N)
        
        # Pass through Graph Transformer
        out = self.graph_transformer(x_lc_enc, mask)  # (N, contrastive_dim)
        
        return out, batch
