import torch
import copy
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class Transformer(Module):
    """A transformer model. User is able to modified the attributes as needed.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    Examples::
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # self._init_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        r"""Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the mask for the src sequence (optional).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the encoder output (optional).
        Shape:
            src: :math:`(S, N, E)`
            tgt: :math:`(T, N, E)`
            src_mask: math:`(S, S)`
            tgt_mask: :math:`(T, T)`
            memory_mask: :math:`(T, S)`
            Note: The masked positions are filled with float('-inf'). 
                  Unmasked positions are filled with float(0.0). Masks ensure that the predictions 
                  for position i depend only on the information before position i.
            output: :math:`(T, N, E)` 
            Note: Due to the multi-head attention architecture in the transformer model, 
                  the output sequence length of a transformer is same as the input sequence
                  (i.e. target) length of the decode. 
            where S is the source sequence length, T is the target sequence length,
            N is the batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

    def generate_square_subsequent_mask(self, n):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(n, n)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask    

    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required). 
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None):
        r"""Pass the input through the endocder layers in turn.
        Args:
            src: the sequnce to the encoder (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required). 
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask, memory_mask)

        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None):
        r"""Pass the input through the endocder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm1(src)
        src = src + self.dropout1(self.self_attn(src2, src2, src2, key_padding_mask=src_mask)[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.dropout1(self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask)[0])
        tgt2 = self.norm2(tgt)

        tgt = tgt + self.dropout2(self.multihead_attn(tgt2, memory, memory, key_padding_mask=memory_mask)[0])
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])