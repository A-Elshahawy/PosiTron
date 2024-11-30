import torch

# from .models import make_model
from factories.ModelFactory import create_transformer

# First, let's set up some example parameters
src_vocab_size = 10000  # Source vocabulary size
tgt_vocab_size = 8000  # Target vocabulary size

# Create a transformer with absolute positional encoding
transformer = create_transformer(
    pe_type="relative",
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    n_heads=8,
    n_layers=2,
    dropout=0.1,
    max_length=5000,
)

# Example input tensors (simulating source and target sequences)

# Create random input tensors
batch_size = 32
src_seq_length = 50
tgt_seq_length = 60

src = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

# Optional: Create masks (for demonstration)
src_mask = torch.ones(batch_size, 1, src_seq_length, dtype=torch.bool)
tgt_mask = torch.ones(batch_size, 1, tgt_seq_length, dtype=torch.bool)

# Forward pass
output = transformer(src, tgt, src_mask, tgt_mask)

# The output will be the generated sequence probabilities
print("Output shape:", output.shape)
