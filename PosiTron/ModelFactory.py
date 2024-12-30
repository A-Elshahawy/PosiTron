import copy
from abc import ABC, abstractmethod
from typing import Literal, TypeVar

import torch
import torch.nn as nn

from PosiTron.layers.Attention import AbsoluteMultiHeadedAttention as Abs_MHA
from PosiTron.layers.Attention import AliBiMultiHeadAttention as AliBi_MHA
from PosiTron.layers.Attention import RelativeMultiHeadAttention as Rel_MHA
from PosiTron.layers.Attention import RotaryMultiHeadAttention as Rope_MHA
from PosiTron.layers.core import EncoderDecoder, Generator
from PosiTron.layers.Decoder import Decoder, DecoderLayer
from PosiTron.layers.Encoder import Encoder, EncoderLayer
from PosiTron.layers.utils import Embeddings, PositionWiseFeedForward

T = TypeVar("T", bound=nn.Module)
Tensor = torch.Tensor


class TransformerConfig:
    """Configuration class for Transformer hyperparameters."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_length: int = 5000,
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_length = max_length


class AbstractTransformer(ABC, nn.Module):
    """Abstract class for Transformer models."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder_decoder: EncoderDecoder | None = self._build_encoder_decoder()

    @property
    def encode(self):
        """Direct access to encoder method."""
        return self.encoder_decoder.encode

    @property
    def decode(self):
        """Direct access to decoder method."""
        return self.encoder_decoder.decode

    @property
    def generator(self):
        """Direct access to generator method."""
        return self.encoder_decoder.generator

    @abstractmethod
    def _get_attention_classes(self):
        """Return attention classes specific to the transformer type."""
        raise NotImplementedError

    def _initialize_params(self, model: T):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _build_encoder_decoder(self) -> EncoderDecoder:
        """Returns an initialized EncoderDecoder instance."""
        c = copy.deepcopy
        cfg = self.config
        attns = self._get_attention_classes()

        attns_kwargs = {
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "dropout": cfg.dropout,
        }

        # Only add 'max_length' if the attention class supports it
        if hasattr(self.config, "max_length") and attns[0] in [Rel_MHA, Rope_MHA]:
            attns_kwargs["max_length"] = self.config.max_length

        encoder_attn = attns[0](**attns_kwargs)
        decoder_self_attn = attns[1](**attns_kwargs)
        decoder_src_attn = attns[2](**attns_kwargs)

        ff = PositionWiseFeedForward(
            d_ff=cfg.d_ff, d_model=cfg.d_model, dropout=cfg.dropout
        )

        encoder = Encoder(
            EncoderLayer(
                size=cfg.d_model,
                self_attn=c(encoder_attn),
                feed_forward=c(ff),
                dropout=cfg.dropout,
            ),
            N=cfg.n_layers,
        )

        decoder = Decoder(
            DecoderLayer(
                size=cfg.d_model,
                self_attn=c(decoder_self_attn),
                src_attn=c(decoder_src_attn),
                feed_forward=c(ff),
                dropout=cfg.dropout,
            ),
            N=cfg.n_layers,
        )

        src_embed = Embeddings(cfg.d_model, cfg.src_vocab_size)
        tgt_embed = Embeddings(cfg.d_model, cfg.tgt_vocab_size)
        generator = Generator(cfg.d_model, cfg.tgt_vocab_size)

        # create encoder-decoder
        encoder_decoder = EncoderDecoder(
            encoder, decoder, src_embed, tgt_embed, generator
        )

        # initialize parameters
        self._initialize_params(encoder_decoder)

        return encoder_decoder

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the transformer."""
        output = self.encoder_decoder(src, tgt, src_mask, tgt_mask)
        return self.encoder_decoder.generator(output)


class AbsoluteTransformer(AbstractTransformer):
    """Transformer with Absolute Positional Encoding."""

    def _get_attention_classes(self):
        return Abs_MHA, Abs_MHA, Abs_MHA


class RelativeTransformer(AbstractTransformer):
    """Transformer with Relative Positional Encoding."""

    def _get_attention_classes(self):
        return Rel_MHA, Rel_MHA, Rel_MHA


class RotaryTransformer(AbstractTransformer):
    """Transformer with Rotary Positional Encoding."""

    def _get_attention_classes(self):
        return Rope_MHA, Rope_MHA, Rope_MHA


class AliBiTransformer(AbstractTransformer):
    """Transformer with AliBi Positional Encoding."""

    def _get_attention_classes(self):
        return AliBi_MHA, AliBi_MHA, AliBi_MHA


class TransformerFactory:
    _transformer_type = {
        "absolute": AbsoluteTransformer,
        "relative": RelativeTransformer,
        "rotary": RotaryTransformer,
        "alibi": AliBiTransformer,
    }

    @classmethod
    def _create(
        cls,
        pe_type: Literal["absolute", "relative", "rotary", "alibi"],
        src_vocab_size: int,
        tgt_vocab_size: int,
        **kwargs,
    ) -> AbstractTransformer:
        """
        Create a transformer of specified type.

        Args:
            pe_type: Position encoding type
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            **kwargs: Additional configuration parameters

        Returns:
            Configured transformer model
        """

        if pe_type not in cls._transformer_type:
            raise ValueError(f"Invalid position encoding type: {pe_type}")

        config = TransformerConfig(
            src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, **kwargs
        )
        return cls._transformer_type[pe_type](config)


def create_transformer(
    pe_type: Literal["absolute", "relative", "rotary"],
    src_vocab_size: int,
    tgt_vocab_size: int,
    **kwargs,
) -> object:
    """Convenience function for creating transformers."""
    return TransformerFactory._create(pe_type, src_vocab_size, tgt_vocab_size, **kwargs)
