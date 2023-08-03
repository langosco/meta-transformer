"""
scratch code for VAE
"""
class Encoder(nn.Module):
    latent_dim: int
    hidden_sizes: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x):
        hidden_sizes = self.hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [int(self.latent_dim * 1.5)]

        for hidden_size in hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class Decoder(nn.Module):
    output_dim: int
    hidden_sizes: Optional[Sequence[int]] = None

    @nn.compact
    def __call__(self, x):
        hidden_sizes = self.hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [int(x.shape[-1] * 1.5)]

        for hidden_size in hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x