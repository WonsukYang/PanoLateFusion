import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, embed_dim, num_heads, n_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embed_scale = math.sqrt(embed_dim)
        self.pos_embed = 
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            layer = TransformerEncoder(embed_dim,
                                       n_heads,
                                       dropout)
            self.layers.append(layer) 
    
    def forward(self, q, k, v):
        for layer in self.layers:
            x = layer(x, k, v)

class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):

class Attention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        assert embed_dim % num_head == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.embed_scale = embed_dim ** -0.5
        self.linear = nn.ModuleDict({
            'q': nn.Linear(embed_dim, embed_dim, bias=True),
            'v': nn.Linear(embed_dim, embed_dim, bias=True),
            'k': nn.Linear(embed_dim, embed_dim, bias=True),
        })

    def forward(self, query, key, value):
        ''' query.size = bs * seq_len * embed_dim
        '''

        batch_size, seq_len, feature_dim = query.size()

        # batch_size, seq_len, embed_dim
        q = self.linear['q'](query)
        k = self.linear['k'](key)
        v = self.linear['v'](value) 
        
        q = q.contiguous()
             .transpose(0, 1)
             .view(seq_len, batch_size * self.num_heads, self.head_dim)
             .transpose(0, 1)
        k = k.contiguous()
             .transpose(0, 1)
             .view(-1, batch_size * self.num_heads, self.head_dim)
             .transpose(0, 1)
        v = v.contiguous()
             .transpose(0, 1)
             .view(-1, batch_size * self.num_heads, self.head_dim)
             .transpose(0, 1)

        attn = torch.bmm(q, k.tranpose(1, 2)) * self.embed_scale
        attn = F.softmax(attn.float(), dim=-1)
        attn = self.dropout(attn)

        attn = torch.bmm(attn, v).view(batch_size, seq_len, embed_dim)


        
