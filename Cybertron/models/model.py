import copy
from typing import TypeVar

import torch
import torch.nn as nn
from layers.Attention import AbsoluteMultiHeadedAttention as Abs_MHA
from layers.core import EncoderDecoder, Generator
from layers.Decoder import Decoder, DecoderLayer
from layers.Encoder import Encoder, EncoderLayer
from layers.utils import Embeddings, PositionWiseFeedForward
from PositionEncondings.absolutePE import AbsolutePositionEncoding as Abs_PE

T = TypeVar("T", bound=nn.Module)
type Tensor = torch.Tensor


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
    att = Abs_MHA(h=h, d_model=d_model)
    ff = PositionWiseFeedForward(d_ff=d_ff, d_model=d_model, dropout=dropout)
    position_encoding = Abs_PE(d_model=d_model, dropout=dropout)

    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                size=d_model, self_Attn=c(att), feed_forward=c(ff), dropout=dropout
            ),
            N=N,
        ),
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
