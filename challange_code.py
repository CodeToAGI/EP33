"""
CodeToAGI — Deep Learning Series EP33 Challenge
Build a 2-Layer Transformer Encoder from scratch in pure PyTorch.

Requirements:
1. pip install torch
2. Implement TransformerEncoderBlock
3. Stack 2 blocks into TransformerEncoder
4. Create batch of 8 random sequences, length 20, d_model=128
5. Forward pass → print output shape (should be torch.Size([8, 20, 128]))
6. Count total trainable parameters
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention + residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward + residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # src: (batch, seq_len)
        x = self.pos_enc(self.embed(src))  # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Challenge setup
    batch_size = 8
    seq_len = 20
    d_model = 128
    vocab_size = 5000

    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        n_heads=8,
        d_ff=4 * d_model,  # classic 4× expansion
    )

    # Random token IDs
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        out = model(src)

    print("Input shape :", src.shape)
    print("Output shape:", out.shape)          # Expected: torch.Size([8, 20, 128])

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Quick scaling insight
    print("\nHow params scale with d_model:")
    print("  - Embeddings ≈ vocab × d_model")
    print("  - Each MultiHeadAttention ≈ 4 × d_model²")
    print("  - Each FFN ≈ 2 × d_model × (4×d_model) = 8 × d_model²")
    print("  → Dominant term is O(d_model²) per layer")
