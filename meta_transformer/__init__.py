import os
from meta_transformer.meta_model import MetaModelClassifier
from meta_transformer.transformer import Transformer
from meta_transformer.vit import VisionTransformer

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
on_cluster = "SCRATCH" in os.environ or "SLURM_CONF" in os.environ
