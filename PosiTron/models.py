from model_factory import (
    AbsoluteTransformer,
    AliBiTransformer,
    RelativeTransformer,
    RotaryTransformer,
    TransformerConfig,
)

# Default configs
DEFAULT_SRC_VOCAB = 5000
DEFAULT_TGT_VOCAB = 5000


# Pre-configured transformer classes
class OptimusPrime(AbsoluteTransformer):
    def __init__(
        self, src_vocab=DEFAULT_SRC_VOCAB, tgt_vocab=DEFAULT_TGT_VOCAB, **kwargs
    ):
        config = TransformerConfig(src_vocab, tgt_vocab, **kwargs)
        super().__init__(config)


class Bumblebee(AliBiTransformer):
    def __init__(
        self, src_vocab=DEFAULT_SRC_VOCAB, tgt_vocab=DEFAULT_TGT_VOCAB, **kwargs
    ):
        config = TransformerConfig(
            src_vocab, tgt_vocab, d_model=256, n_heads=4, **kwargs
        )
        super().__init__(config)


class Megatron(RelativeTransformer):
    def __init__(
        self, src_vocab=DEFAULT_SRC_VOCAB, tgt_vocab=DEFAULT_TGT_VOCAB, **kwargs
    ):
        config = TransformerConfig(
            src_vocab, tgt_vocab, d_model=1024, n_heads=16, **kwargs
        )
        super().__init__(config)


class Starscream(RotaryTransformer):
    def __init__(
        self, src_vocab=DEFAULT_SRC_VOCAB, tgt_vocab=DEFAULT_TGT_VOCAB, **kwargs
    ):
        config = TransformerConfig(src_vocab, tgt_vocab, n_layers=12, **kwargs)
        super().__init__(config)
