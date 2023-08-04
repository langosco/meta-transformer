from typing import Optional
import jax
from meta_transformer.transformer import Transformer
import flax.linen as nn
import einops


class Patches(nn.Module):
  """A module that extracts patches from an image and flattens them."""
  patch_size: int
  embed_dim: int

  def __call__(
      self,
      image_batch: jax.Array,  # [B, H, W, C]
  ) -> jax.Array:  # [B, T, D]
    conv = nn.Conv2D(
        self.embed_dim,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='VALID'
    )
    patches = conv(image_batch)  # [B, H', W', D]
    b, h, w, d = patches.shape
    #return jnp.reshape(patches, [b, h * w, d])
    return einops.rearrange("b h w d -> b (h w) d", patches)


class VisionTransformer(nn.Module):  # Untested
  """A ViT-style classifier."""

  transformer: Transformer
  model_size: int
  num_classes: int
  name: Optional[str] = None
  patch_size: int = 4

  def __call__(
      self,
      image_batch: jax.Array,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass. Returns a sequence of logits."""
    extract_patches = Patches(patch_size=self.patch_size, embed_dim=self.model_size)
    patches = extract_patches(image_batch)  # [B, T, D]
    _, seq_len, _ = patches.shape

    # Embed the patches and positions.
    embed_init = nn.initializers.variance_scaling(
      scale=0.2, distribution="truncated_normal")
    embedding = nn.Dense(self.model_size, kernel_init=embed_init)
    patch_embedding = embedding(patches)  # [B, T, D]

    positional_embeddings = self.param(
        'positional_embeddings',
        embed_init,
        (seq_len, self.model_size)
    )
    input_embeddings = patch_embedding + positional_embeddings  # [B, T, D]

    # Run the transformer over the inputs.
    embeddings = self.transformer(
        input_embeddings,
        is_training=is_training,
    )  # [B, T, D]
    
    first_out = embeddings[:, 0, :]  # [B, V]
    return nn.Dense(self.num_classes, name="linear_output")(first_out)  # [B, V]
