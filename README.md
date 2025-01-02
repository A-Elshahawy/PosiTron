# PosiTron: An Educational Library for Transformer Models and Positional Encodings

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<<<<<<< HEAD
**PosiTron** is a Python library designed as an educational resource for exploring Transformer models and various positional encoding mechanisms. Inspired by the concept of "position" in transformers, and the dynamic and powerful "-tron" from the Transformers franchise, this library provides a hands-on approach to learning about these complex architectures.
=======
**PosiTron** : is a Python library designed as an educational resource for exploring Transformer models and various positional encoding mechanisms. Inspired by the concept of "position" in transformers, and the dynamic and powerful "-tron" from the Transformers franchise, this library provides a hands-on approach to learning about these complex architectures.
>>>>>>> backup-master

**This library is intended for educational purposes only.**

## Key Learning Points

This library aims to help users learn about:

* **Transformer Architectures:** Understand the building blocks of Transformer models (encoder, decoder, attention).
* **Multi-Headed Attention:** Explore the mechanisms of scaled dot-product attention.
* **Positional Encodings:** Experiment with various positional encoding methods:
  * Absolute Positional Encoding
  * Relative Positional Encoding
  * Rotary Positional Encoding (RoPE)
  * Attention with Linear Biases (ALiBi)
* **Implementation Details:** Learn how these concepts are implemented in code.
* **Model Factory**: Understand how the model factory is implemented to provide many options of model creations.
* **Modular and Flexible Model Creation**: See how this implementation is done.

## Core Features

* **Diverse Positional Encodings:** Implements multiple positional encoding methods in a flexible and modular way.
* **Pre-configured Transformer Models:** Includes pre-configured models like `OptimusPrime`, `Bumblebee`, `Megatron `and `Starscream`, each with different positional encodings and default settings.
* **Modular Components:**  Provides separate classes for attention mechanisms, positional encodings, layers, and model creation.
* **Clear Code:** Designed for readability and ease of understanding.
* **Well-documented code:** Designed to be easy to learn.
* **Direct Imports:** Supports importing transformer classes and models directly for maximum customization and understanding.

## Quick Start

### Installation

```bash
# This is just to try the code, this is for educational use case,
# there is no package at pypi yet
# pip install positron
```

Clone this repo into your local machine to start experimenting with different positional encoding mechanisms.

### Usage Examples

#### 1. Importing and Using Pre-Configured Transformer Models

```python
from models import OptimusPrime, Bumblebee, Megatron, Starscream

# Create instances with default configurations
optimus = OptimusPrime()
megatron = Megatron()
bee= Bumblebee()
starscream = Starscream()
```

#### 2. Creating Models With Custom Configurations

```python
from models import OptimusPrime, Bumblebee, Megatron, Starscream
from ModelFactory import TransformerConfig

# Using TransformerConfig
config = TransformerConfig(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    n_heads=8,
    n_layers=6,
)
optimus = OptimusPrime(config=config)

# Using keyword arguments for Bumblebee with different n_heads and layers
Bumblebee = Bumblebee(n_heads=2, n_layers=3)
```

#### 3. Creating Models via Factory from ModelFactory

```python
from ModelFactory import create_transformer

<<<<<<< HEAD
alibi_model = create_transformer(pe_type='alibi', d_model=256, n_heads=4, src_vocab_size=1000, tgt_vocab_size=1000)
absolute_model = create_transformer(pe_type='absolute', d_model=512, n_heads=8, src_vocab_size=1000, tgt_vocab_size=1000)
relative_model = create_transformer(pe_type='relative', d_model=512, n_heads=8, max_length=5000, src_vocab_size=1000, tgt_vocab_size=1000)
rotary_model = create_transformer(pe_type='rotary', d_model=512, n_heads=8, max_length=5000, src_vocab_size=1000, tgt_vocab_size=1000)
=======
bumblebee = create_transformer(pe_type='alibi', d_model=256, n_heads=4, src_vocab_size=1000, tgt_vocab_size=1000)
optimus = create_transformer(pe_type='absolute', d_model=512, n_heads=8, src_vocab_size=1000, tgt_vocab_size=1000)
megatron = create_transformer(pe_type='relative', d_model=512, n_heads=8, max_length=5000, src_vocab_size=1000, tgt_vocab_size=1000)
starcream = create_transformer(pe_type='rotary', d_model=512, n_heads=8, max_length=5000, src_vocab_size=1000, tgt_vocab_size=1000)
>>>>>>> backup-master
```

## Package Structure

```
positron/
├── main.py          # main to test different positional encodings
├── ModelFactory.py     # Central factory class for creating different Transformer models with PE
├── models.py           #  Pre-configured Transformer classes, using factory from ModelFactory
├── layers/            # Contains core transformer building blocks
│   ├── Attention.py    # Classes for different attention mechanisms
│   ├── core.py        # EncoderDecoder class and Generator for final output
│   ├── Decoder.py      # Classes for the decoder
│   ├── Encoder.py      # Classes for the encoder
│   └── utils.py         # Utility modules like Embeddings and feed-forward
├── PositionEncodings/ # Contains various positional encodings implementations
│   ├── absolutePE.py   # Class for Absolute PE
│   ├── alibi.py        # Class for ALiBi PE
│   ├── relativePE.py  # Class for relative PE
│   └── rotaryPE.py     # Class for RoPE
│
└── train/             # Training utilities for your models
    ├── inference.py   # Utilities for performing model inference
    ├── loss.py        # Loss functions
    ├── trainer.py     # Training logic
    └── utils.py       # Utility functions

```

## Inspiration

This project was inspired by the following key works:

* **The Annotated Transformer:** [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
  * A detailed explanation of the Transformer architecture.
* **Attention is All You Need:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
  * The original paper that introduced the Transformer.
* **RoFormer: Enhanced Transformer with Rotary Position Embedding:** [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
  * Introduces Rotary Position Encoding.
* **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation:** [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)
  * Introduces ALiBi positional encoding.
* **Self-Attention with Relative Position Representations:** [https://arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)
  * Introduced the relative positional embeddings.

## Contributing

Contributions to the PosiTron library are welcome, especially if they are related to educational purposes.

MIT License

## Contact

<<<<<<< HEAD
* **LinkedIn:** [Ahmed Elshahawy](linkedin.com/in/ahmed-elshahawy-a42149218)
* **Gmail:** [Ahmed ELshahawy ](ahmedelshahawy078@gmail.ocm)
=======
You can find me on :[
](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)

* [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmed-elshahawy-a42149218/)

* [![Gmail](https://img.shields.io/badge/Gmail-Email-red?style=flat&logo=gmail)](mailto:ahmedelshahawy078@gmail.com)
>>>>>>> backup-master
