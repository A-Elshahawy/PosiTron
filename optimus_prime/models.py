import copy
from typing import TypeAlias, TypeVar

import torch
import torch.nn as nn

from .Attention import MultiHeadedAttention as MHA
from .Decoder import Decoder, DecoderLayer
from .Encoder import Encoder, EncoderLayer
from .General import EncoderDecoder, Generator
from .position_embeddeding import AbsolutePositionEncoding, Embeddings
from .utils import Position_Wise_FeedForward

T = TypeVar("T", bound=nn.Module)
Tensor: TypeAlias = torch.Tensor


def make_model(
    src_vocab: Tensor,
    tgt_vocab: Tensor,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> T:
    "helper function: Construct a Model from hyperparams"
    c = copy.deepcopy
    att = MHA(h=h, d_model=d_model)
    ff = Position_Wise_FeedForward(d_ff=d_ff, d_model=d_model, dropout=dropout)
    position_encoding = AbsolutePositionEncoding(d_model=d_model, dropout=dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(att), c(ff), dropout=dropout), N=N),
        Decoder(
            DecoderLayer(
                size=d_model,
                self_attn=c(att),
                src_attn=c(att),
                feed_forward=c(ff),
                dropout=dropout,
            ),
            N=N,
        ),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position_encoding)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position_encoding)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
