import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # heads for Q
    n_kv_heads: Optional[int] = None  # heads for K, V
    vocab_size: int = -1  # will set later
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None


class RotaryPositionEmbedding(nn.Module):
    def __init__(self):
        super(RotaryPositionEmbedding, self).__init__()
        pass

    def forward(self, x):
        pass


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # Learnable parameter for scaling
        self.gemma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: Input tensor of shape (Batch_Size, SeqLen, d_model) # d_model is the embedding dimension
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdims=True)+self.eps)

        normalized_x = (x/rms_x)*self.gemma

        return normalized_x


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Determine the number of key-value heads (defaults to n_heads if not specified)
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads

        # Set the number of query heads and the number of repetitions for K and V
        self.n_heads_q = config.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # Calculate the head dimension
        self.head_dim = config.dim // config.n_heads

        # Linear transformations for queries, keys, values, and output
        self.Wq = nn.Linear(config.dim, config.n_heads *
                            self.head_dim, bias=False)
        self.Wk = nn.Linear(config.dim, config.n_kv_heads *
                            self.head_dim, bias=False)
        self.Wv = nn.Linear(config.dim, config.n_kv_heads *
                            self.head_dim, bias=False)
        self.Wo = nn.Linear(config.n_heads * self.head_dim,
                            config.dim, bias=False)

        # Initialize key and value caches with zeros
        self.cache_k = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.n_kv_heads, self.head_dim))

        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(
            self.head_dim, config.max_seq_len, config.device)

    @staticmethod
    def repeat_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:

        # Repeat the heads of K and V to match the number of heads in Q

        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        else:
            return (x[:, :, :, None, :]
                    .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                    .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
                    )

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape  # (B, 1, dim)
        assert dim == self.dim, "dim must be equal to self.dim"

        # (B, 1, dim) -> (B, 1, n_heads_q * head_dim)
        xq = self.Wq(x)

        # (B, 1, dim) -> (B, 1, n_kv_heads * head_dim)
        xk = self.Wk(x)

        # (B, 1, dim) -> (B, 1, n_kv_heads * head_dim)
        xv = self.Wv(x)

        # (B, 1, n_heads_q * head_dim) -> (B, 1, n_heads_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # (B, 1, n_kv_heads * head_dim) -> (B, 1, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = self.rope(xq, start_pos)
        xk = self.rope(xk, start_pos)

        # Update key and value caches
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrieve key and value caches
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # Repeat the heads of K and V to match the number of heads in Q
        keys = self.repeat_heads(keys, self.n_rep)
        values = self.repeat_heads(values, self.n_rep)

        # (B, 1, n_heads_q, head_dim) -> (B, n_heads_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, n_heads_q, 1, head_dim) * (B, n_heads_q, head_dim, SeqLen) -> (B, n_heads_q, 1, SeqLen)
        scores = torch.matmul(xq, keys.transpose(-2, -1)
                              ) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, n_heads_q, 1, SeqLen) * (B, n_heads_q, SeqLen, head_dim) -> (B, n_heads_q, 1, head_dim)
        context = torch.matmul(scores, values)

        # (B, n_heads_q, 1, head_dim) -> (B, 1, head_dim)
        context = context.transpose(
            1, 2).contiguous().view(batch_size, seq_len, -1)

        # (B, 1, head_dim) -> (B, 1, dim)
        output = self.Wo(context)

        return output


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Calculate the hidden dimension based on the provided parameters
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)

        # Adjust the hidden dimension based on ffn_dim_multiplier (if provided)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        # Ensure hidden_dim is a multiple of config.multiple_of
        hidden_dim = config.multiple_of * \
            ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        # Define linear layers for the feedforward network
        self.fc1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.fc3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch_Size, SeqLen, Dim)

        # Apply the first linear transformation and activation (swish)
        swish = F.silu(self.fc1(x))

        # Apply the second linear transformation
        x_V = self.fc3(swish)

        # Element-wise multiplication
        x = swish * x_V

        # Apply the third linear transformation
        x = self.fc2(x)

        return x  # Return the output


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super(EncoderBlock, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.num_heads

        self.attention = SelfAttention(config)
        self.feed_forward = FeedForward(config)

        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Transformer, self).__init__()

        # Check if vocab_size is specified
        assert config.vocab_size != -1, "vocab_size must be specified"

        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.embeddings = nn.Embedding(self.vocab_size, config.d_model)

        self.layers = nn.ModuleList([EncoderBlock(config)
                                    for _ in range(self.n_layers)])

        self.norm = RMSNorm(config, config.norm_eps)

        self.output = nn.Linear(config.d_model, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # Input shape: (Batch_Size, SeqLen)

        # Ensure seq_len is 1
        assert x.shape[1] == 1, "seq_len must be 1"

        # Embedding lookup
        x = self.embeddings(x)

        # Pass through each transformer encoder block
        for layer in self.layers:
            x = layer(x, start_pos)

        # Layer normalization
        x = self.norm(x)

        # Output prediction
        x = self.output(x)

        return x  # Return the output
