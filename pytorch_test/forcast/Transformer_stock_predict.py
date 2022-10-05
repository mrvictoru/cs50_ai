# Class for transformer model and related functions such as positional encoding and dataset

import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.d_model = d_model # dimension of the ouput of sub-layers in the model
        self.dropout = nn.Dropout(p=dropout) # dropout rate
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        # adapt from PyTorch tutorial
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        if torch.cuda.is_available():
            position = position.cuda()
            div_term = div_term.cuda()
        
        if self.batch_first:
            pe = torch.zeros(1, max_len, d_model)
            pe[0,:,0::2] = torch.sin(position * div_term)
            pe[0,:,1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, d_model)
            pe[:,0,0::2] = torch.sin(position * div_term)
            pe[:,0,1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, d_model] or
                [enc_seq_len, batch_size, d_model]
        
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1),:]
        else:
            x = x + self.pe[:x.size(0),:,:]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
        input_size: int,                        # number of input variables/features. 1 if univariate
        dec_seq_len: int,                       # the length of input sequence fed to the decoder
        batch_first: bool,                      # whether the input is [batch_size, seq_len, input_size] or [seq_len, batch_size, input_size]
        dim_val: int = 512,                     # aka d_model. All sub-layers in the model produce outputs of dimension d_model
        num_encoder_layers: int = 4,            # number of stacked encoder layers in the encoder
        num_decoder_layers: int = 4,            # number of stacked decoder layers in the decoder
        num_heads: int = 8,                     # number of heads in the multi-head attention (parallel attention layers)
        dropout_encoder: float = 0.1,           # the dropout rate of the encoder
        dropout_decoder: float = 0.1,           # the dropout rate of the decoder
        dropout_pos_encoder: float = 0.1,       # the dropout rate of the positional encoder
        dim_feedforward_encoder: int = 2048,    # the dimension of the feedforward network model in the encoder/number of neurons in the linear layer of the encoder
        dim_feedforward_decoder: int = 2048,    # the dimension of the feedforward network model in the decoder/number of neurons in the linear layer of the decoder
        num_predicted_features: int = 1,        # number of features to predict
        activation: str = 'relu',
        device: str = 'cpu'):

        super().__init__()
        self.dec_seq_len = dec_seq_len
        # create the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(in_features = input_size, out_features = dim_val)
        self.decoder_input_layer = nn.Linear(in_features = num_predicted_features, out_features = dim_val)
        self.linear_mapping = nn.Linear(in_features = dim_val, out_features = num_predicted_features)

        # create positional encoder
        self.pos_encoder = PositionalEncoder(d_model = dim_val, dropout = dropout_pos_encoder, batch_first = batch_first)

        # create encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model = dim_val, nhead = num_heads, dim_feedforward = dim_feedforward_encoder, dropout = dropout_encoder, activation = activation, batch_first = batch_first)
        # stack the encoder layers, normalization is none as it is handle by TransformerEncoderLayer by default
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_encoder_layers, norm = None)
        
        # create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model = dim_val, nhead = num_heads, dim_feedforward = dim_feedforward_decoder, dropout = dropout_decoder, activation = activation, batch_first = batch_first)
        # stack the decoder layers, normalization is none as it is handle by TransformerDecoderLayer by default
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_decoder_layers, norm = None)

    def forward(self, src: Tensor, tgt: Tensor, src_mask:Tensor=None, tgt_mask: Tensor = None) -> Tensor:
        """
        Return a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:
            src:    the encoder's output sequence. Shape: (S,E) for unbatched input,
                    (S,N,E) if batch_first = False or (N,S,E) if batch_first = True,
                    where S is the source sequence length, N is the batch size, and E is the number of features (1 if univariate)

            tgt:    the sequence to the decoder. Shape: (T,E) for unbatched input,
                    (T,N,E) if batch_first = False or (N,T,E) if batch_first = True,
                    where T is the target sequence length, N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from using data points from the target sequence
        """
        # encode the input sequence
        # pass through the input layer right before the encoder
        src = self.encoder_input_layer(src)
        # pass through the positional encoding layer
        src = self.pos_encoder(src)
        # pass through all the stacked encoder layers in the encoder
        # masking is not needed as input sequences are of the same length
        src = self.encoder(src)

        # decode the encoded sequence
        tgt = self.decoder_input_layer(tgt)

        # pass through all the stacked decoder layers in the decoder
        tgt = self.decoder(tgt, src, tgt_mask, src_mask)

        # map the output to the desired output size
        output = self.linear_mapping(tgt)

        return output

# Dataset class used for transformer models
class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.tensor, indices: list, enc_seq_len: int, dec_seq_len: int, target_seq_len: int) -> None:
        """
        Args:
            data:   tensor, the entire train, validation or test data squences
                    before any slicing. if univariate the shape will be [num_samples, num_features]
                    where the number of feature will be 1 + number of exogenous variables.
                    Number of exogenous variables would be 0 if univariate

            indices:    list of tuples, Each tuple has two elements:
                        1) the start index of a sub-sequence
                        2) the end index of a sub-sequence
                        The sub-sequence is split into src, trg and trg_y later

            enc_seq_len: int, length of the encoder sequence
            dec_seq_len: int, length of the decoder sequence
            target_seq_len: int, length of the target sequence
        """
        self.data = data
        self.indices = indices
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len
