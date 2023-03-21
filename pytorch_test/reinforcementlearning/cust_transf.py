"""
This module contains the classes for creating a custom transformer based model for acting as a decision making agent for the trading environment.
Based on https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py

"""

# import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the masked attention class
class MaskedAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.n_heads = n_heads
        self.drop_p = drop_p

        # feed forward layers which create the query, key and value vectors
        self.query = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.value = nn.Linear(h_dim, h_dim)

        # feed forward layer which project the attention to the correct dimension
        self.proj = nn.Linear(h_dim, h_dim)

        # dropout layer
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        # create the mask
        mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)

        # register_buffer will make the mask a constant tensor
        # so that it will not be included in the model parameters and be updated during backpropagation
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        # x: [batch_size, T, h_dim]
        B, T, C = x.shape # batch size, sequence length, hidden dimension * number of heads
        N, D = self.n_heads, C // self.n_heads # number of heads, dimension of each head

        # compute the query, key and value vectors
        q = self.query(x).view(B, T, N, D).transpose(1, 2) # [batch_size, n_heads, T, D]
        k = self.key(x).view(B, T, N, D).transpose(1, 2) # [batch_size, n_heads, T, D]
        v = self.value(x).view(B, T, N, D).transpose(1, 2) # [batch_size, n_heads, T, D]

        # compute the attention
        weights = Q @ K.transpose(2,3) / math.sqrt(D) # QK^T / sqrt(D)
        weights = weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # mask the future tokens
        normalized_weights = F.softmax(weights, dim=-1) # softmax along the last dimension
        A = self.att_drop(normalized_weights @ v) # attention with dropout

        # compute the output
        # gather heads and project to the correct dimension
        attention = A.transpose(1, 2).contiguous().view(B, T, N*D)
        out = self.proj_drop(self.proj(attention))

        return out
    
# define the attention block with layer normalization and residual connection as well as the feed forward layer
class AttentionBlock(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedAttention(h_dim, max_T, n_heads, drop_p)
        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.forward = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p)
        )

    def forward(self, x):
        # x: [batch_size, T, h_dim]
        # attention -> layer normalization -> residual connection -> feed forward -> layer normalization -> residual connection
        out = self.norm1(x + self.attention(x))
        out = self.norm2(out + self.forward(out))
        return out
    
# define the decision transformer model
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep = 4096):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        input_seq_len = 3 * context_len
        blocks = [AttentionBlock(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection heads (project to embedding dimension)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)

        #
