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
        out.update({f"val/{k}": v for k, v in aux["metrics"].items()})
        return out, aux

    @functools.partial(jit, static_argnums=0)
    def init_params(self, rng: ArrayLike, data: dict) -> dict:
        """Initializes state of the updater."""
        out_rng, subkey = jax.random.split(rng)
        params = self.model.init(subkey, data["input"], is_training=False)
        params = params.unfreeze()  # necessary to work with optax.MultiTransform
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
        }
        metrics.update({f"train/{k}": v for k, v in aux["metrics"].items()})
        metrics.update({
            f"opt/{k}": v for k, v in state.opt_state.hyperparams.items()
            })
        state.step += 1
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = random.split(state.rng)
        metrics, aux = self.get_metrics_and_loss(subkey, state.params, data)
        return state, metrics, aux


def print_metrics(metrics: Mapping[str, Any], prefix: str = ""):
    """Prints metrics to stdout. Assumption: metrics is a dict of scalars
    and always contains the keys "step" and "epoch".
    """
    print(prefix, end="")
    print(f"Step: {metrics['step']}, Epoch: {metrics['epoch']}, ", end="")
    try:
        metrics_to_print = ["train/loss", "grad_norm"]
        print(", ".join([f"{k}: {metrics[k]:.3f}" for k in metrics_to_print]))
    except KeyError:
        metrics_to_print = [k for k in metrics.keys() if k not in ["step", "epoch"]]
        print(", ".join([f"{k}: {round(metrics[k], 3)}" for k in metrics_to_print]))


@chex.dataclass
class Logger:
    log_interval: int = 50
    store: bool = False

    def __post_init__(self):
        self.data = []

    def clean(self, metrics: dict) -> dict:
        metrics = {k: v.item() if isinstance(v, jnp.ndarray) else v
                for k, v in metrics.items()}
        return metrics

    def log(self,
            state: TrainState,
            metrics: dict,
            force_log: Optional[bool] = False):
        metrics = self.clean(metrics)
        if state.step % self.log_interval == 0 or force_log:
            print_metrics(metrics)
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

