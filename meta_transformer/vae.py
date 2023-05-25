@dataclasses.dataclass
class Encoder(hk.Module):
    latent_dim: int
    hidden_sizes: Optional[Sequence[int]] = None

    def __call__(self, x):
        if self.hidden_sizes is None:
            self.hidden_sizes = [int(self.latent_dim * 1.5)]
        layers = []
        for hidden_size in self.hidden_sizes:
            layers.append(hk.Linear(hidden_size))
            layers.append(jax.nn.gelu)
        layers.append(hk.Linear(self.latent_dim))
        return hk.Sequential(layers)(x)


@dataclasses.dataclass
class Decoder(hk.Module):
    output_dim: int
    hidden_sizes: Optional[Sequence[int]] = None

    def __call__(self, x):
        if self.hidden_sizes is None:
            self.hidden_sizes = [int(x[-1] * 1.5)]
        layers = []
        for hidden_size in self.hidden_sizes:
            layers.append(hk.Linear(hidden_size))
            layers.append(jax.nn.gelu)
        layers.append(hk.Linear(self.output_dim))
        return hk.Sequential(layers)(x)
