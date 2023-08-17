from typing import Optional
import flax.linen as nn
import jax
import chex
from meta_transformer.attention import SelfAttention
from meta_transformer.utils import get_activation_stats


def mup_dense_scaling(scale: float = 1.0):
    """Scale dense weights with variance proportional to 1 / d."""
    return nn.initializers.variance_scaling(
        scale=scale/6,  # originally 2 / self.num_layers
        mode="fan_in",
        distribution="truncated_normal",
    )

mup_attn_scaling = mup_dense_scaling


class TransformerBlock(nn.Module):
    num_heads: int
    d_model: int  # d_model = num_heads * key_size
    dropout_rate: float
    widening_factor: Optional[int] = 4
    attn_factor: Optional[float] = 1.0
    init_scale: Optional[float] = 1.0
    name: Optional[str] = None

    def setup(self):
        self.self_attention = SelfAttention(
            num_heads=self.num_heads,
            kernel_init=mup_attn_scaling(self.init_scale),
            name="self_attention",
            attn_factor=self.attn_factor,
        )

    def dense(self, features: int, name: Optional[str] = None):
        return nn.Dense(
            features, kernel_init=mup_dense_scaling(self.init_scale), name=name)

    @nn.compact
    def __call__(
        self,
        x,
        mask: jax.Array = None,
        is_training: bool = True,
    ) -> jax.Array:
        activations = dict()

        residual = x
        activations["residual_pre_attention"] = get_activation_stats(residual)

        x = nn.LayerNorm()(x)
        x, acts = self.self_attention(x, mask=mask)  # can include mask=mask argument here
        activations["attention"] = acts
        activations["attention_out"] = get_activation_stats(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        x = x + residual

        residual = x
        activations["residual_pre_mlp"] = get_activation_stats(residual)

        x = nn.LayerNorm()(x)
        x = self.dense(self.widening_factor * self.d_model)(x)
        activations["mlp_mid"] = get_activation_stats(x)
        x = nn.gelu(x)
        x = self.dense(self.d_model)(x)
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
    attn_factor: float = 1.0
    init_scale: Optional[float] = 1.0
    name: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,  # [batch, seq, dim] (just [seq, dim] also works I think).
        is_training: bool = True,
    ) -> jax.Array:
        """Transforms input embedding sequences to output embedding sequences."""
        chex.assert_shape(x, (None, None, self.d_model))
        activations = {}

        for layer in range(self.num_layers):
            x, acts = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                widening_factor=self.widening_factor,
                attn_factor=self.attn_factor,
                init_scale=self.init_scale,
            )(x, is_training=is_training)
            activations[f"layer_{layer}"] = acts

        return nn.LayerNorm()(x), activations