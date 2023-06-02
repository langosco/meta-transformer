import jax
from jax import random, jit, value_and_grad, nn
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import chex
import functools
from typing import Mapping, Any, Tuple, List, Iterator, Optional, Dict
from jax.typing import ArrayLike
from meta_transformer import utils, preprocessing, torch_utils, module_path
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
import wandb
from nninn.repl.utils import load_nets, classes_per_task
import nninn
import os
import argparse
from dataclasses import asdict


# Optimizer and update function
@chex.dataclass
class TrainState:
    step: int
    rng: random.PRNGKey
    opt_state: optax.OptState
    params: dict


@chex.dataclass(frozen=True)  # needs to be immutable to be hashable (for static_argnums)
class Updater: # Could also make this a function of loss_fn, model.apply, etc if we want to be flexible
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation
    model: hk.TransformedWithState
    loss_fn: callable

    # TODO this is hardcoded to val
    def get_metrics_and_loss(self, rng, params, data):
        """Compute acc and loss on test set."""
        loss, metrics = self.loss_fn(params, rng, data, is_training=False)
        out = {"val/loss": loss}
        out.update({f"val/{k}": v for k, v in metrics.items()})
        return out

    @functools.partial(jit, static_argnums=0)
    def init_params(self, rng: ArrayLike, data: dict) -> dict:
        """Initializes state of the updater."""
        out_rng, subkey = jax.random.split(rng)
        params = self.model.init(subkey, data["input"])
        opt_state = self.opt.init(params)
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = jax.random.split(state.rng)
        (loss, aux_metrics), grads = value_and_grad(self.loss_fn, has_aux=True)(
                state.params, subkey, data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        metrics = {
                "train/loss": loss,
                "step": state.step,
        }
        metrics.update({f"train/{k}": v for k, v in aux_metrics.items()})
        state.step += 1
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = random.split(state.rng)
        return state, self.get_metrics_and_loss(subkey, state.params, data)


@chex.dataclass
class Logger:
    log_interval: int = 50
    disable_wandb: bool = False
    store: bool = False

    def __post_init__(self):
        self.data = []

    def clean(self, train_metrics, val_metrics):
        metrics = train_metrics or {}

        if val_metrics is not None:
            metrics.update(val_metrics)

        metrics = {k: v.item() if isinstance(v, jnp.ndarray) else v
                   for k, v in metrics.items()}
        return metrics

    # TODO change args, eg metrics, optional_metrics
    def log(self,
            state: TrainState,
            train_metrics: dict = None,
            val_metrics: dict = None):
        metrics = self.clean(train_metrics, val_metrics)
        if state.step % self.log_interval == 0 or val_metrics is not None:
            print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            if not self.disable_wandb:
                wandb.log(metrics, step=state.step)
            if self.store:
                self.log_append(state, metrics)
    
    def log_append(self, 
                   state: TrainState,
                   metrics: dict):
        metrics['step'] = int(state.step)
        self.data.append(metrics)
    
    def get_data(self, metric="train/loss"):
        """returns a tuple (steps, metric)"""
        return zip(*[(d['step'], d[metric]) for d in self.data if metric in d])

