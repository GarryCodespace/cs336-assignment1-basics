import math

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Apply a learned linear transformation to the final input dimension."""

    def __init__(self, in_features: int, out_features: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """Look up a learned vector for each input token ID."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize the final dimension by its RMS and apply a learned gain."""

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * inv_rms * self.weight).to(in_dtype)


class SiLU(nn.Module):
    """Apply the elementwise activation x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Gated feed-forward layer with a SiLU activation and no biases."""

    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        # Round the default hidden dimension up to a multiple of 64.
        self.d_ff = 64 * math.ceil(8 * d_model / (3 * 64)) if d_ff is None else d_ff
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize scores after subtracting their maximum for stability."""
    exp_scores = (x - x.amax(dim=dim, keepdim=True)).exp()
    return exp_scores / exp_scores.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Attend to keys allowed by a boolean mask, with arbitrary batch axes."""
    compute_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
    scores = (Q.to(compute_dtype) @ K.to(compute_dtype).transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        # An entirely masked query returns zero rather than producing NaNs.
        scores = scores.masked_fill(~mask.any(dim=-1, keepdim=True), 0)
    probabilities = softmax(scores, dim=-1)
    if mask is not None:
        probabilities = probabilities.masked_fill(~mask, 0)
    return (probabilities @ V.to(compute_dtype)).to(V.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent query/key coordinates using cached positional angles."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("RoPE requires a positive, even head dimension")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None] * inv_freq
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention with batched projections across all heads."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float = 10000.0, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device) if max_seq_len is not None else None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.num_heads, self.d_head).transpose(-3, -2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # Share each token's position across the attention heads.
            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        merged = attended.transpose(-3, -2).reshape(*x.shape[:-1], self.d_model)
        return self.output_proj(merged)


class TransformerBlock(nn.Module):
    """Pre-norm causal attention and SwiGLU with residual connections."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float = 10000.0, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    """Decoder-only Transformer mapping token IDs to next-token logits."""

    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float = 10000.0, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("Input sequence exceeds the model context length")
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))
