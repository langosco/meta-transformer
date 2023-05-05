import chex
from typing import Optional,Union,Tuple
import jax.random 
import jax.numpy as jnp

@chex.dataclass(frozen=True)
class Parameters:
    seed: jax.random.PRNGKey
    dataset: Optional[str] = "CIFAR10" #cifar10, cifar100 or mnist
    augment: Optional[bool] = False
    num_classes: Optional[int] = 9
    class_dropped: Optional[int] = 8
    model_name: Optional[str] = "lenet5" #smallCNN, largeCNN, lenet5 or alexnet, resnet not available yet because it's stateful
    activation: Optional[str] = "leakyrelu" #relu, leakyrelu,tanh,sigmoid,silu,gelu
    init: Optional[str] = None #U - uniform, N - normal, TN -truncated normal
    data_mean: Optional[Union[Tuple[jnp.float32],jnp.float32]] = (0.5,0.5,0.5)
    data_std: Optional[jnp.float32] = (0.5,0.5,0.5)
    batch_size: Optional[int] = 32
    num_epochs: Optional[int] = 50
    optimizer: Optional[str] = "adamW"
    dropout: Optional[jnp.float32] = 0.5
    weight_decay: Optional[jnp.float32] = 1e-4
    lr: Optional[jnp.float32] = 3e-4
    
def sample_parameters(rng_key, dataset_name, model_name=None,opt=None,num_epochs=None, augment=False):
    new_key, seed, key_class_dropped, key_act, key_init, key_batch,key_dropout, key_weight_decay, key_lr,key_opt,key_model = jax.random.split(rng_key, num=9)
    
    # dataset specific one-class-omission
    if dataset_name == "MNIST":
        num_classes=9
        data_mean=0.5
        data_std=0.5
    elif dataset_name == "CIFAR10":
        num_classes=9
        data_mean = (0.49139968, 0.48215827,0.44653124)
        data_std = (0.24703233, 0.24348505, 0.26158768)
    elif dataset_name == "CIFAR100":
        num_classes=99
        data_mean = (0.49139968, 0.48215827,0.44653124)
        data_std = (0.24703233, 0.24348505, 0.26158768)
    else:
        raise ValueError("Unknown dataset name")
    class_dropped = jax.random.randint(key_class_dropped, (), 0, num_classes)
    
    # activation
    activations = ["relu", "leakyrelu", "tanh", "sigmoid", "silu", "gelu"]
    activation = activations[jax.random.randint(key_act, (), 0, len(activations))]
    
    # init
    inits = [None, "U", "N", "TN"]
    init = inits[jax.random.randint(key_init, (), 0, len(inits))]
    
    # batch
    batch_sizes = [32, 64, 128]
    batch_size = batch_sizes[jax.random.randint(key_batch, (), 0, len(batch_sizes))]
    
    # dropout
    dropout = jax.random.uniform(key_dropout, (), minval=0.0, maxval=0.5)
    
    # weight decay
    log_weight_decay = jax.random.uniform(key_weight_decay, (), minval=-4.0, maxval=-2.0)
    weight_decay = jnp.power(10.0, log_weight_decay)
    
    # learning rate
    log_lr = jax.random.uniform(key_lr, (), minval=-4.0, maxval=-3.0)
    lr = jnp.power(10.0, log_lr)
    
    # optionally fixed parameters
    # optimizer
    if opt == None:
        optimizers = ["adamW","sgd"]
        opt = optimizers[jax.random.randint(key_opt, (), 0, len(optimizers))]
        
    if num_epochs == None:
        num_epochs = 50
        
    if model_name==None:
        models = ["smallCNN", "largeCNN", "lenet5","alexnet"]
        model_name = models[jax.random.randint(key_model,(),0,len(models))]
    
    return new_key,Parameters(seed=seed, 
                      dataset=dataset_name, 
                      augment=augment,
                      num_classes=num_classes, 
                      class_dropped=class_dropped.item(), 
                      model_name=model_name,
                      activation=activation,
                      init=init, 
                      data_mean=data_mean,
                      data_std = data_std,
                      batch_size=batch_size,
                      num_epochs=num_epochs,
                      optimizer=opt,
                      dropout=dropout.item(),
                      weight_decay=weight_decay.item(),
                      lr=lr.item())
    