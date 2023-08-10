"""
A simple base transformer class.
"""

from typing import Optional
from jaxtyping import Array
import flax.linen as nn
import jax
import chex
import jax.numpy as jnp
from meta_transformer.attention import SelfAttention
from meta_transformer.utils import get_activation_stats


# def width_scaling(width_factor):
#     """Initialize weights with stddev 1 / sqrt(d_model)."""
#     def init(key, named_shape, dtype):
#         stddev = 0.02 / jnp.sqrt(width_factor)
#         stddev = stddev / jnp.array(.87962566103423978, dtype)
#         return jax.random.truncated_normal(key, -2, 2, named_shape, dtype) * stddev
#     return init


# For all weights


class TransformerBlock(nn.Module):
    num_heads: int
    d_model: int  # d_model = num_heads * key_size
    dropout_rate: float
    widening_factor: int = 4
    name: Optional[str] = None
    weight_init: Optional[callable] = None
    mup_attn_multiplier: float = 1.0

    @nn.compact
    def __call__(
        self,
        x,
        mask: jax.Array = None,
        is_training: bool = True,
    ) -> jax.Array:
        activations = dict()

        self_attention = SelfAttention(
            num_heads=self.num_heads,
            kernel_init=self.weight_init,
            name="self_attention",
            mup_attn_multiplier=self.mup_attn_multiplier,
        )

        residual = x
        activations["residual_pre_attention"] = get_activation_stats(residual)

        x = nn.LayerNorm()(x)
        x, acts = self_attention(x, mask=mask)  # can include mask=mask argument here
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        activations["attention"] = acts
        activations["attention_out"] = get_activation_stats(x)
        x = x + residual

        residual = x
        activations["residual_pre_mlp"] = get_activation_stats(residual)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.widening_factor * self.d_model, kernel_init=self.weight_init)(x)
        activations["mlp_mid"] = get_activation_stats(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, kernel_init=self.weight_init)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        activations["mlp_out"] = get_activation_stats(x)
        x = x + residual

        return x, activations


class Transformer(nn.Module):
    num_heads: int
    num_layers: int
    d_model: int  # can be inferred from x.shape[-1]
    dropout_rate: float
    widening_factor: int = 4
    name: Optional[str] = None
    weight_init: Optional[callable] = None
    mup_attn_multiplier: float = 1.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [batch, seq, dim] (just [seq, dim] also works I think).
        is_training: bool = True,
    ) -> jax.Array:
        """Transforms input embedding sequences to output embedding sequences."""

#        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        chex.assert_shape(x, (None, None, self.d_model))
        activations = dict()

        for layer in range(self.num_layers):
            x, acts = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                widening_factor=self.widening_factor,
                weight_init=self.weight_init,
                mup_attn_multiplier=self.mup_attn_multiplier,
            )(x, is_training=is_training)
            activations[f"layer_{layer}"] = acts

        return nn.LayerNorm()(x), activations