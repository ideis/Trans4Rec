import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer
from torch.nn.init import xavier_uniform_

class Trans4Rec(Transformer):
    r"""A transformer model applied for sequential recommendations.
    Args:
        n_items: the number of items to recommend.
        d_model: the dimension of the encoder/decoder embedding models (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> transformer_model = nn.Transformer(n_items, nhead=8, num_encoder_layers=6)
    """

    def __init__(self, n_items, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Trans4Rec, self).__init__(d_model=d_model,
                                        nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout)

        self.src_embed = nn.Embedding(n_items, d_model, padding_idx=0)
        self.pos_embed = PositionalEncoding(d_model, dropout)
        self.decoder = nn.Linear(d_model, n_items)

        self.n_items = n_items
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self._init_parameters()

    def forward(self, src, src_mask=None, shared_embeddings=False):
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            src_mask: the mask for the src sequence (optional).
            shared_embeddings: use existing embeddings matrix for decoding instead Linear transformation(optional).
        Shape:
            src: [source sequence length, batch size]
            src_mask: [batch size, source sequence length]
            output: [source sequence length, batch size, src_vocab]
        Examples:
            >>> output = transformer_model(src, src_mask=src_mask)
        """
        src = self.src_embed(src)
        src = self.pos_embed(src)

        memory = self.encoder(src, src_mask)
        if shared_embeddings:
            output = torch.matmul(memory, self.src_embed.weight.data.permute(1, 0))
        else:
            output = self.decoder(memory)
        return output

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens 
        in the sequence. The positional encodings have the same dimension as 
        the embeddings, so that the two can be summed. Here, we use sine and cosine 
        functions of different frequencies.
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]    
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)