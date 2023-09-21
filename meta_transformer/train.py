import numpy as np
import jax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import optax
import chex
import functools
from typing import Mapping, Any, Tuple, List, Iterator, Optional
from jax.typing import ArrayLike
import wandb
import flax.linen as nn
from meta_transformer.data import Data
from meta_transformer.logger_config import setup_logger
logger = setup_logger(__name__)


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
    model: nn.Module
    loss_fn: callable

    # TODO this is hardcoded to val
    def get_metrics_and_loss(self, rng, params, data):
        """Compute acc and loss on test set."""
        loss, aux = self.loss_fn(params, rng, data, is_training=False)
        out = {"val/loss": loss}
        if "metrics" in aux:
            out.update({f"val/{k}": v for k, v in aux["metrics"].items()})
        return out, aux

    @functools.partial(jit, static_argnums=0)
    def init_train_state(self, rng: ArrayLike, data: Data) -> dict:
        out_rng, subkey = jax.random.split(rng)
        v = self.model.init(subkey, data.input, is_training=False)  # TODO hardcoded to detection (change to 'inputs'?)
        opt_state = self.opt.init(v["params"])
        if list(v.keys()) != ["params"]:
            raise ValueError("Expected model.init to return a dict with "
                f"a single key 'params'. Instead got {v.keys()}.")
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=v["params"],
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data: Data) -> (TrainState, dict):
        state.rng, subkey = jax.random.split(state.rng)
        (loss, aux), grads = value_and_grad(self.loss_fn, has_aux=True)(
                state.params, subkey, data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        metrics = {
                "train/loss": loss,
                "step": state.step,
                "grad_norm": optax.global_norm(grads),
                "weight_norm": optax.global_norm(state.params),
#                "diff_to_input": jnp.mean((data.target - aux["outputs"])**2)  # TODO hardcoded to detection
        }
        if "metrics" in aux:
            metrics.update({f"train/{k}": v for k, v in aux["metrics"].items()})
        metrics.update({
            f"opt/{k}": v for k, v in state.opt_state.hyperparams.items()
            })
        state.step += 1
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict) -> (TrainState, dict):
        state.rng, subkey = random.split(state.rng)
        metrics, aux = self.get_metrics_and_loss(subkey, state.params, data)
        return state, metrics, aux


def print_metrics(metrics: Mapping[str, Any], prefix: str = ""):
    """Prints metrics to stdout. Assumption: metrics is a dict of scalars
    and always contains the keys "step" and "epoch".
    """
    for k, v in metrics.items():
        metrics[k] = np.round(v.item(), 7)
    output = prefix
    output += f"Step: {metrics['step']}, Epoch: {metrics['epoch']}, "
    other_metrics = [k for k in metrics if k not in ["step", "epoch"]]
    output += ", ".join([f"{k}: {metrics[k]:.4f}" for k in other_metrics])
    logger.info(output)


@chex.dataclass
class Logger:
    def __post_init__(self):
        self.train_metrics = []
        self.val_metrics = []
    
    def flush_mean(self, state, status="train", verbose=True, 
                   extra_metrics=None):
        metrics = self.train_metrics if status == "train" else self.val_metrics
        if len(metrics) == 0:
            raise ValueError(f"No metrics currently logged for status={status}.")

        # reduce
        means = {}
        for k in metrics[0].keys():
            means[k] = np.mean([d[k] for d in metrics])
        means["step"] = int(state.step)

        # update
        if extra_metrics is not None:
            means.update(extra_metrics)
        
        # log
        wandb.log(means, step=state.step)
        if verbose:
            print_metrics(means)
        
        # reset
        if status == "train":
            self.train_metrics = []
        elif status == "val":
            self.val_metrics = []
        else:
            raise ValueError(f"Unknown status {status}.")

    def write(self, state, metrics, status="train"):
        metrics["step"] = int(state.step)
        if status == "train":
            self.train_metrics.append(metrics)
        elif status == "val":
            self.val_metrics.append(metrics)
        else:
            raise ValueError(f"Unknown status {status}.")
    
    def get_metrics(self, metric="train/loss"):
        """returns a tuple (steps, metric)"""
        return zip(*[(d['step'], d[metric]) for d in self.train_metrics if metric in d])

