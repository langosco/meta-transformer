"""
A simple base transformer class.
"""

from typing import Optional
import flax.linen as nn
import jax
import chex


attn_default_init = nn.initializers.variance_scaling(
    scale=2.0 / 24 * 0.4**2, # !?!
#    scale=0.02,
    mode="fan_in",  # change?
    distribution="truncated_normal",
)

dense_default_init = nn.initializers.variance_scaling(
#    scale=0.3**2,  # !?
    scale=0.02,
    mode="fan_in",  # change?
    distribution="uniform",
)

class TransformerBlock(nn.Module):
    num_heads: int
    d_model: int  # d_model = num_heads * key_size
    dropout_rate: float
    widening_factor: int = 4
    name: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x,
        mask: jax.Array = None,
        is_training: bool = True,
    ) -> jax.Array:

        # TODO: since hk implementation used to divide by num_layers (typically 24),
        # I've divided by 24 here as well. I'll keep it for the first test run to
        # properly compare against the hk implementation, but remove it afterwards.
#        init_fn = lambda rng, shape, dtype: nn.initializers.variance_scaling(
#            scale=2.0/24,
#            mode="fan_in",  # change?
#            distribution="truncated_normal",
#        )(rng, shape, dtype) * 0.87962566103423980   # Lauro: multiply by .88 to reduce std to match haiku's default

        self_attention = nn.SelfAttention(
            num_heads=self.num_heads,
            kernel_init=attn_default_init,
        )

        residual = x
        x = nn.LayerNorm()(x)
        x = self_attention(x, mask=mask)  # can include mask=mask argument here
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.widening_factor * self.d_model, kernel_init=dense_default_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.d_model, kernel_init=dense_default_init)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)
        x = x + residual

        return x


class Transformer(nn.Module):
    num_heads: int
    num_layers: int
    d_model: int  # can be inferred from x.shape[-1]
    dropout_rate: float
    widening_factor: int = 4
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

        for _ in range(self.num_layers):
            x = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                dropout_rate=self.dropout_rate,
                widening_factor=self.widening_factor,
            )(x, is_training=is_training)

        return nn.LayerNorm()(x)
