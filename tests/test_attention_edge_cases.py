import pytest
import torch
import torch.nn.functional as F

from cs336_basics.model import (
    MultiHeadSelfAttention,
    RotaryPositionalEmbedding,
    TransformerLM,
    scaled_dot_product_attention,
)


@pytest.mark.parametrize("masked", [False, True])
def test_attention_outputs_and_gradients_match_pytorch(masked):
    generator = torch.Generator().manual_seed(17)
    inputs = [
        torch.randn(*shape, generator=generator, dtype=torch.float64).requires_grad_()
        for shape in [(2, 3, 4, 6), (2, 3, 5, 6), (2, 3, 5, 7)]
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    mask = None
    if masked:
        mask = torch.rand(4, 5, generator=generator) > 0.3
        mask[0] = False
    actual = scaled_dot_product_attention(*inputs, mask)
    expected = F.scaled_dot_product_attention(*reference_inputs, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    if masked:
        torch.testing.assert_close(actual[..., 0, :], torch.zeros_like(actual[..., 0, :]))
    upstream = torch.randn(actual.shape, generator=generator, dtype=actual.dtype)
    actual.backward(upstream)
    expected.backward(upstream)
    for x, reference in zip(inputs, reference_inputs):
        torch.testing.assert_close(x.grad, reference.grad)


def test_attention_avoids_half_precision_score_overflow():
    q = torch.full((2, 3, 4), 1000.0, dtype=torch.float16)
    k = q.clone()
    v = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
    actual = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).half()
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected)


def test_rope_uses_positions_and_preserves_vector_norms():
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=4, max_seq_len=8)
    x = torch.ones(2, 3, 4)
    positions = torch.tensor([[0, 1, 3], [2, 4, 7]])
    # For input pairs (1, 1), rotation yields (cos - sin, sin + cos).
    angles = positions.float()[..., None] * torch.tensor([1.0, 0.01])
    expected = torch.stack((angles.cos() - angles.sin(), angles.sin() + angles.cos()), dim=-1).flatten(-2)
    actual = rope(x, positions)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))
    assert not list(rope.parameters())
    assert not rope.state_dict()


@pytest.mark.parametrize("with_rope", [False, True])
def test_multihead_attention_preserves_causal_prefixes(with_rope):
    torch.manual_seed(18)
    attention = MultiHeadSelfAttention(8, 2, max_seq_len=8 if with_rope else None)
    x = torch.randn(2, 6, 8)
    changed = x.clone()
    changed[:, 3:] += 100
    with torch.no_grad():
        full = attention(x)
        torch.testing.assert_close(full[:, :3], attention(changed)[:, :3])
        torch.testing.assert_close(full[:, :3], attention(x[:, :3]))


def test_multihead_rope_supports_multiple_batch_axes_and_custom_positions():
    torch.manual_seed(19)
    attention = MultiHeadSelfAttention(8, 2, max_seq_len=10)
    x = torch.randn(2, 3, 4, 8)
    positions = torch.tensor([
        [[0, 1, 3, 5], [1, 3, 4, 7], [2, 4, 6, 8]],
        [[0, 2, 5, 7], [1, 2, 3, 9], [3, 4, 5, 8]],
    ])
    with torch.no_grad():
        batched = attention(x, positions)
        for i in range(2):
            for j in range(3):
                torch.testing.assert_close(batched[i, j], attention(x[i, j], positions[i, j]))


def test_transformer_lm_causality_backward_and_state_roundtrip():
    torch.manual_seed(20)
    model = TransformerLM(32, 8, 8, 2, 2, 16)
    tokens = torch.randint(0, 32, (2, 6))
    logits = model(tokens)
    assert logits.shape == (2, 6, 32)
    changed = tokens.clone()
    changed[:, 3:] = (changed[:, 3:] + 1) % 32
    with torch.no_grad():
        torch.testing.assert_close(logits[:, :3], model(changed)[:, :3])
        torch.testing.assert_close(logits[:, :3], model(tokens[:, :3]))
    F.cross_entropy(logits.flatten(0, 1), tokens.flatten()).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    restored = TransformerLM(32, 8, 8, 2, 2, 16)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(logits, restored(tokens))
    with pytest.raises(ValueError, match="context length"):
        model(torch.zeros(2, 9, dtype=torch.long))
