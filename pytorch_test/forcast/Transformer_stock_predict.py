# Class for transformer model and related functions such as positional encoding and dataset, etc.
# see https://github.com/KasperGroesLudvigsen/influenza_transformer for reference
from collections.abc import Sequence
import torch
import torch.nn as nn
import math
from torch import nn, Tensor
from typing import Tuple, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

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

            enc_seq_len: int, length of input sequence given to the encoder layer of the transformer model.

            dec_seq_len: int, length of input sequence given to the decoder layer of the transformer model.

            target_seq_len: int, length of the target sequence (the output of the model)

            target_idx: the index position of the target variable in data. (Data is a 2D tensor)
        """
        super().__init__()
        self.data = data
        self.indices = indices
        print("From get_src_trg: data size = {}".format(data.size()))

        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target sequence)
        """
        # get teh first element of the idx-th tuple in indices
        start_idx = self.indices[idx][0]
        # get the second (and last) element of the idx-th tuple in indices
        end_idx = self.indices[idx][1]

        sequence = self.data[start_idx:end_idx]

        src,trg,trg_y = self.get_src_trg(sequence, self.enc_seq_len, self.dec_seq_len, self.target_seq_len)
        
        return src, trg, trg_y
    
    def get_src_trg(self,sequence:torch.Tensor, enc_seq_len: int, dec_seq_len: int, target_seq_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target) sequences from a sequence.

        Args:
            sequence: tensor, a 1D tensor of length n where n = encoder input length + target sequence length
            enc_seq_len: int, length of input sequence given to the encoder layer of the transformer model.
            target_seq_len: int, desired length of the target sequence (the one against which the model output is compared)
        Returns:
            src: tensor, 1D, input to the model
            trg: tensor, 1D, input to the model
            trg_y: tensor, the target sequence against which  the model output is compared when calculating the loss
        """
        # get the encoder input sequence
        src = sequence[:enc_seq_len]
        # get the decoder input sequence (same dimension as the target sequence, must contain the last value of src, and all values of trg_y except the last, i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        # get the target sequence
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "trg_y length does not match target_seq_len"

        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]

# utility function to create the indices for the train, validation and test datasets
def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates a mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    see https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:
        dim1: int, the length of the first dimension of the mask, for both src and tgt masking, this must be target sequence length
        dim2: int, the length of the second dimension of the mask, for src masking, this must be the encoder sequence length (length of input sequence to the model), for tgt masking, this must be the target sequence length
    
    Return:
        Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)

def get_indices_input_target(num_obs, input_len, step_size, forecast_horizon, target_len):
    """
    Produce all the start and end index position of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which the modle will be trained.

    Args:
        num_obs:    int, the number of observations in the dataset for which indices must be generated

        input_len:  int, the length of the input sequence to the model (a sub-sequence of the entire data sequence)

        step_size:  int, the step size between two consecutive sub-sequences
                    if 1, the first sub-sequence will be indices 0-input_len, and the next will be 1-input_len.

        forecast_horizon:   int, Number of index positions that the target is away from the last index position of the input sequence
                            if horizon is 1, and the input sequence is data [0:10], the target will be data [11:target_len]

        target_len: int, the length of the target sequence / output sequence
    
    Returns a tuple with 4 elements:
    1) The index positions of the first element to be inlcuded in the input sequence
    2) The index positions of the last element to be included in the input sequence
    3) The index positions of the first element to be included in the target sequence
    4) The index positions of the last element to be included in the target sequence
    """
    input_len = round(input_len) # just a precaution
    start_position = 0
    stop_position = num_obs - 1

    inpseq_first_idx = start_position
    inpseq_last_idx = inpseq_first_idx + input_len
    target_first_idx = inpseq_last_idx + forecast_horizon
    target_last_idx = target_first_idx + target_len
    print("target_last_idx = {}".format(target_last_idx))
    print("stop_position = {}".format(stop_position))
    indices = []
    while target_last_idx <= stop_position:
        indices.append((inpseq_first_idx, inpseq_last_idx, target_first_idx, target_last_idx))
        inpseq_first_idx += step_size
        inpseq_last_idx += step_size
        target_first_idx += inpseq_last_idx + forecast_horizon
        target_last_idx += target_first_idx + target_len
    return indices

def get_indices_entire_sequence(data: pd.Dataframe, window_size: int, step_size: int) -> list:
    """
    Produce all the start and end index position of all sub-sequences.
    The indices will be used to split the data into sub-sequences on which the modle will be trained.

    Args:
        data:   pd.Dataframe, the dataset for which indices must be generated

        window_size:    int, the desired length of the sub-sequences. Should be (input_len + target_len)
                        E.g. if you want the model to consider the past 100 time steps in order to preidct the future 50 time steps, window_size should be 100+50=150
        
        step_size:  int, the step size between two consecutive sub-sequences
                    if 1, the first sub-sequence will be [0:window_size], and the next will be [1:window_size].

    Returns:
        list of tuples, each tuple contains the start and end index position of a sub-sequence.
        These tuples should be used to slice the dataset into sub-sequences. These sub-sequences should then be passed into a function
        that slices them into input and target sequences. 
    """
    stop_position = len(data)-1 # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = subseq_first_idx + window_size

    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
    return indices

def to_numeric_and_downcast_data(df: pd.DataFrame):
    """
    Downcast columns in df to smallest possible version of it's existing data
    type
    """
    fcols = df.select_dtypes('float').columns
    
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df

def get_stock_data_yf(stock_name, past_years, interval) -> pd.DataFrame:
    """
    Get stock data from yahoo finance

    Args:
        stock_name: str, the name of the stock to get data for
        past_years: int, the number of years of data to get
        interval:   str, the interval at which the data is collected. Valid values are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
                    see https://pypi.org/project/yfinance/ for more details
    
    Returns:
        pd.DataFrame, the stock data
    """
    data = yf.download(stock_name, period=f'{past_years}y', interval=interval)
    data = to_numeric_and_downcast_data(data)
    return data