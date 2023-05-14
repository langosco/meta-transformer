import functools
import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from typing import Optional, Mapping, Any, Dict, Tuple
from jax.typing import ArrayLike
import datasets
import dm_pix as pix


def load_data(dataset_name: str = "mnist"):
    """Load mnist, cifar10 or cifar100 dataset."""
    dataset = datasets.load_dataset(dataset_name)
    dataset = dataset.with_format("jax")
    if dataset_name == "mnist":
        dataset = dataset.rename_column("image", "img")
    train, test = dataset["train"], dataset["test"]
    train_images, train_labels = train["img"], train["label"]
    test_images, test_labels = test["img"], test["label"]
    return train_images, train_labels, test_images, test_labels


def random_rot90(rng, image):
    """randomly rotate the image 50% of the time"""
    rot = random.bernoulli(rng, 0.5)
    return jax.lax.cond(
        rot,
        lambda img: pix.rot90(img, 1),
        lambda img: img,
        image
    )


def random_resized_crop(rng, image):
    """Randomly crop and resize an image."""
    # TODO doesn't work bc of the random shape of the crop
    k0, k1 = random.split(rng)
    delta = random.randint(k0, (2,), 0, 5)
    image = pix.random_crop(k1, image, (32 - delta[0], 32 - delta[1], 3))
    image = jax.image.resize(image, (32, 32, 3), method="bicubic")
    return image


def augment_datapoint(rng, img):
    """Apply a random augmentation to a single image. Pixel values are assumed to be in [0, 1]"""
    rng = random.split(rng, 7)
#    img = pix.random_brightness(rng[0], img, 0.3)
#    img = pix.random_contrast(rng[1], img, lower=0.2, upper=3)
#    img = pix.random_saturation(rng[2], img, lower=0, upper=3)
    img = pix.random_flip_left_right(rng[2], img)
    img = pix.random_flip_up_down(rng[3], img)
    img = random_rot90(rng[4], img)
    return img


def process_datapoint(rng: jnp.ndarray, 
                      img: jnp.array,
                      augment: bool = True) -> jnp.array:
    img = img / 255.0
    img = jax.lax.cond(  # Random augment?
            augment, 
            lambda img: augment_datapoint(rng, img),
            lambda img: img,
            img
        )
    return img


@functools.partial(jit, static_argnums=2)
def process_batch(rng, batch, augment = True):
    """Apply a random augmentation to a batch of images.
    Input is assumed to be a jnp.array of shape (B, H, W, C) with 
    values in [0, 255]."""
    rng = random.split(rng, len(batch))
    proc = functools.partial(process_datapoint, augment=augment)
    return vmap(proc)(rng, batch)