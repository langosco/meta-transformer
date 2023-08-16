"""
A simple base transformer class.
"""

from typing import Optional
from jaxtyping import Array
import flax.linen as nn
import jax
import chex
import jax.numpy as jnp
import numpy as np
from meta_transformer.attention import SelfAttention
from meta_transformer.utils import get_activation_stats



def mup_dense_scaling(scale: float = 1.0):
    """Scale dense weights with variance proportional to 1 / d."""
    return nn.initializers.variance_scaling(
        scale=scale*2,
        mode="fan_in",
        distribution="truncated_normal",
    )


def mup_attn_scaling(scale: float = 1.0):
    """Scale dense weights with variance proportional to 1 / d."""
    return nn.initializers.variance_scaling(
        scale=scale*2,
        mode="fan_in",
        distribution="truncated_normal",
    )


class TransformerBlock(nn.Module):
    num_heads: int
    d_model: int  # d_model = num_heads * key_size
    dropout_rate: float
    widening_factor: Optional[int] = 4
    attn_scale: Optional[float] = 1.0
    init_scale: Optional[float] = 1.0
    name: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x,
        mask: jax.Array = None,
        is_training: bool = True,
    ) -> jax.Array:
        activations = dict()

        def dense(features: int, name: Optional[str] = None):
            return nn.Dense(features, 
                            kernel_init=mup_dense_scaling(), 
                            name=name)

        self_attention = SelfAttention(
            num_heads=self.num_heads,
            kernel_init=mup_attn_scaling(self.init_scale),
            name="self_attention",
            attn_scale=self.attn_scale,
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
        x = dense(self.widening_factor * self.d_model)(x)
        activations["mlp_mid"] = get_activation_stats(x)
        x = nn.gelu(x)
        x = dense(self.d_model)(x)
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
    attn_scale: float = 1.0
    init_scale: Optional[float] = 1.0
    name: Optional[str] = None

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
                attn_scale=self.attn_scale,
                init_scale=self.init_scale,
            )(x, is_training=is_training)
            activations[f"layer_{layer}"] = acts

        return nn.LayerNorm()(x), activations