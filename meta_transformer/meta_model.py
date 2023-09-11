import dataclasses
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from meta_transformer.transformer import Transformer, mup_dense_scaling
from jax.typing import ArrayLike


# output is initialized as 1 / d**2, but we can equivalently set
# init to 0.
mup_output_scaling = nn.initializers.zeros


def input_scaling(scale: float = 1.0):
    return nn.initializers.variance_scaling(
        scale=scale*0.76,
        mode="fan_in",
        distribution="truncated_normal",
    )


@dataclasses.dataclass
class MetaModel(nn.Module):
    """A meta-model that returns neural network parameters."""
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.05
    widening_factor: int = 4
    name: Optional[str] = None
    use_embedding: Optional[bool] = True
    in_factor: Optional[float] = 1.0
    out_factor: Optional[float] = 1.0
    attn_factor: Optional[float] = 1.0
    init_scale: Optional[float] = 1.0

    @nn.compact
    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True,
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        _, seq_len, input_size = inputs.shape

        if self.use_embedding:
            embedding = nn.Dense(self.d_model,
                                 kernel_init=input_scaling(self.init_scale),
                                 name="input/embedding")
            inputs = embedding(inputs)

        inputs = inputs + self.param(
            'input/positional_embeddings',
            nn.initializers.zeros,
            (seq_len, self.d_model)
        )

        inputs = inputs * self.in_factor

        transformer = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            widening_factor=self.widening_factor,
            attn_factor=self.attn_factor,
            init_scale=self.init_scale * 24 / self.num_layers,
            name="transformer",
        )

        outputs, activation_stats = transformer(inputs, is_training=is_training)

        if self.use_embedding:
            unembedding = nn.Dense(
                input_size, 
                kernel_init=mup_output_scaling,
                name="output/unembedding"
            )
            outputs = unembedding(outputs)

        outputs = outputs * self.out_factor
        activation_stats["output"] = jax.numpy.abs(outputs).mean()
        return outputs, activation_stats


class MetaModelClassifier(MetaModel):
    """Binary classifier with scalar output."""
    use_embedding: bool = False

    @nn.compact
    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True,
        ) -> jax.Array:
        inputs = nn.Dense(self.d_model,
                          kernel_init=input_scaling(self.init_scale),
                          name="input/embedding")(inputs)
        outputs, activation_stats = super().__call__(
            inputs, is_training=is_training)
#        outputs = jnp.mean(outputs, axis=1)
        outputs = outputs[:, 0, :]
        outputs = nn.Dense(1, 
                kernel_init=mup_output_scaling, name="output/unembedding")(outputs)
        return jnp.squeeze(outputs), activation_stats


param_labels = {
    "input/embedding": "input",
    "input/positional_embeddings": "input",
    "transformer": "hidden",
    "output/unembedding": "output",
}


def mup_adamw(
        lr_in: float,
        lr_hidden: float,
        lr_out: float,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        wd_in: float = 1e-4,
        wd_hidden: float = 1e-4,
        wd_out: float = 1e-4,
    ):
    opt = optax.multi_transform(
        {
            "input": optax.adamw(lr_in, b1=b1, b2=b2, eps=eps, weight_decay=wd_in),
            "hidden": optax.adamw(lr_hidden, b1=b1, b2=b2, eps=eps, weight_decay=wd_hidden),
            "output": optax.adamw(lr_out, b1=b1, b2=b2, eps=eps, weight_decay=wd_out),
        },
        param_labels
    )
    return opt