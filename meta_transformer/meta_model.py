import dataclasses
from typing import Optional
import jax
import flax.linen as nn
import optax

from meta_transformer.transformer import Transformer#, dense_default_init
from jax.typing import ArrayLike


# for the input embedding:
fan_in_scaling = nn.initializers.variance_scaling(
    scale=1.0,
    mode="fan_in",
    distribution="truncated_normal",
)


@dataclasses.dataclass
class MetaModel(nn.Module):
    """A meta-model that returns neural network parameters."""

    d_model: int
    num_heads: int
    num_layers: int
    dropout_rate: int
    widening_factor: int = 4
    name: Optional[str] = None
    use_embedding: Optional[bool] = True
    in_factor: Optional[float] = 1.0
    out_factor: Optional[float] = 1.0

    @nn.compact
    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True,
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        _, seq_len, input_size = inputs.shape

        embedding = nn.Dense(
            self.d_model, 
            kernel_init=fan_in_scaling,
            name="input/embedding"
        )

        if self.use_embedding:
            inputs = embedding(inputs)

        _, seq_len, _ = inputs.shape

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
            name="transformer",
        )

        outputs, activation_stats = transformer(inputs, is_training=is_training)

        unembedding = nn.Dense(
            input_size, 
            kernel_init=nn.initializers.zeros,
            name="output/unembedding"
        )

        if self.use_embedding:
            outputs = unembedding(outputs)
        return outputs * self.out_factor, activation_stats


param_labels = {
    "params": {
        "input/embedding": "input",
        "input/positional_embeddings": "input",
        "transformer": "hidden",
        "output/unembedding": "output",
    }
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


#######################################
#### unused code below this point #####
#######################################

@dataclasses.dataclass
class MetaModelClassifier(nn.Module):
    """A simple meta-model."""
    num_classes: int
    d_model: int
    num_heads: int
    num_layers: int
    dropout_rate: int
    widening_factor: int = 4
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    @nn.compact
    def __call__(
        self,
        inputs: ArrayLike,  # dict
        *,
        is_training: bool = True,
    ) -> jax.Array:
        """Forward pass. Returns a sequence of logits."""
        #net_embed = NetEmbedding(embed_dim=self.d_model)
        #inputs = vmap(net_embed)(inputs)
        if self.use_embedding:
            inputs = nn.Dense(self.transformer.d_model, kernel_init=fan_in_scaling)(inputs)
        _, seq_len, _ = inputs.shape

        positional_embeddings = self.param(
            'positional_embeddings',
            nn.initializers.zeros(),
#            nn.initializers.normal(stddev=0.02),  # From BERT
            (seq_len, self.d_model)
        )
        inputs = inputs + positional_embeddings  # [B, T, D]

        # Run the transformer over the inputs.
        # Run the transformer over the inputs.
        transformer, _ = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            widening_factor=self.widening_factor,
        )
        outputs = transformer(inputs, is_training=is_training)

        first_out = outputs[:, 0, :]  # [B, D]
        # TODO: this is not the muP way
        return nn.Dense(self.num_classes, kernel_init=fan_in_scaling, name="linear_output")(first_out)  # [B, V]