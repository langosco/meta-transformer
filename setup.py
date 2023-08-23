from setuptools import setup

setup(name='meta-transformer',
      version='0.0.1',
      packages=["meta_transformer"],
      install_requires=[
        "chex",
        "datasets",
        "dm-pix",
        "jax",
        "matplotlib",
        "numpy",
        "optax",
        "pandas",
        "wandb",
        "pytest",
        "torch",
        "torchvision",
        "pillow",
        "einops",
        "orbax-checkpoint",
        "flax"
        #"gen_models"
        #"nnaugment"
        ],
    )
