"""
Adapted for jax models from: https://github.com/HSG-AIML/NeurIPS_2021-Weight_Space_Learning
"""
import itertools
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import math
import numpy as np
import time

from typing import List
from functools import partial

def permute_batch(rng, checkpoints,
                  permute_layers:List[str]=["cnn/conv2_d_1", "cnn/linear"],
                  permutation_mode: str = "random",
                  num_permutations: int = 50,
                  keep_original:bool=True,
                  models_same: bool=True):
    
    batch_size = len(checkpoints)
    if num_permutations is None:
        permutation_mode = 'complete'
    if models_same:
        permutation_list, rng = __get_permutations_per_layer(rng, checkpoints[0], permute_layers, 
                                                        permutation_mode=permutation_mode,
                                                        num_permute = num_permutations*batch_size)
        permutation_list, rng = __get_permutation_combinations(rng, permutation_list,
                                                        permutation_mode=permutation_mode,
                                                        num_permute = num_permutations*batch_size)
    else:
        raise NotImplementedError("Batches of different models cannot yet be permuted as a batch")
    
    checkpoint_list = []
    if keep_original:
        checkpoint_list = checkpoint_list + checkpoints
    repeated_list = [checkpoint for _ in range(num_permutations) for checkpoint in checkpoints]
    checkpoint_list = checkpoint_list + perform_batch_permutation(repeated_list, permutation_list)
    #for i in range(len(permutation_list)//batch_size):
    #    if (i+1)*batch_size < len(permutation_list):
    #        permutations = permutation_list[i*batch_size:(i+1)*batch_size]
    #        batched_result = perform_batch_permutation(checkpoints, permutations)
    #    else:
    #        permutations = permutation_list[i*batch_size:]
    #        batched_result = perform_batch_permutation(checkpoints[:len(permutations)], permutations)
    #    checkpoint_list = checkpoint_list + batched_result
    return checkpoint_list

def permute_checkpoint(rng, checkpoint, 
                       permute_layers:List[str]=["cnn/conv2_d_1", "cnn/linear"],
                       permutation_mode: str = "random",
                       num_permutations: int = 50,
                       keep_original:bool=True):
    """
    Augment one model parameter pytree, by permuting layers specified in permute_layers list
    
    Arguments:
        checkpoint (dict):      jax pytree - which model parameters to permute
        permute_layers (list):  which layers to permute, default: ["cnn/conv2_d_1", "cnn/linear"]
        permutation_mode (str): "random" or "complete" - weather to choose random permutations 
                                to perform or to perform all possible permutations, default: "random"
        num_permutations (int): how many random permutations to perform, ignored if permutation_mode="complete",
                                default=50
        keep_original (bool):   weather to keep original checkpoint as the first element of the output list (True) 
                                or to ommit it (False), defautl: True
    Returns:
        list model checkpoints (as jax pytrees) with permuted weights 
    """
    
    if num_permutations is None:
        permutation_mode = 'complete'
    permutation_list, rng = __get_permutations_per_layer(rng, checkpoint, permute_layers, 
                                                    permutation_mode=permutation_mode,
                                                    num_permute = num_permutations)
    permutation_list, rng = __get_permutation_combinations(rng, permutation_list,
                                                    permutation_mode=permutation_mode,
                                                    num_permute = num_permutations)
    checkpoint_list = []
    if keep_original:
        checkpoint_list.append(checkpoint)
    for i in range(len(permutation_list)):
        check1 = perform_single_permutation(checkpoint, permutation_list[i])
        checkpoint_list.append(check1)
    return checkpoint_list

def __get_permutations_per_layer(rng, checkpoint, permute_layers, permutation_mode='random', num_permute=100):
    """For each layer in permute_layers (named layers in checkpoint parameter dict), get a list of 
    permuted indices. Returns a dictionary with layer names as keys and lists of permuted indices as values"""
    
    permutations = {layer: [] for layer in permute_layers}
    
    for layer in permute_layers:
        # get the dimension along which to permute
        if "conv" in layer:
            w = checkpoint[layer]['w']
            kernel = w.shape[3] #out channel dimension 
        elif "linear" in layer:
            w = checkpoint[layer]['w']
            kernel=w.shape[1] # out features dimension
        else:
            raise ValueError("permutations for layers of this type are not available")
        # generate lists of permuted indices
        index_old = np.arange(kernel)
        if permutation_mode == "complete":
            # save all possible permutations
            for index_new in itertools.permutations(index_old, kernel):
                permutations[layer].append(np.array(index_new))
        elif permutation_mode == "random":
            # save a fixed number of permutations
            num_permute_l = max(1,min(num_permute//len(permute_layers), 
                                  __approximate_num_permutations(kernel)))
            index_new = index_old
            for i in range(num_permute_l):
                rng, subkey = random.split(rng, 2)
                index_new = random.permutation(subkey, index_new)
                permutations[layer].append(np.array(index_new))
        
    return permutations, rng      

def __approximate_num_permutations(n):
    """Approximate n!"""
    return int(round(math.sqrt(2*math.pi*n) * (n/math.e)**n))+1

def __get_permutation_combinations(rng, layer_permutations, permutation_mode='random', num_permute=100):
    """Given a list of permutations per layer, combine them into joint permutations of the whole model
    (randomly seleced num_permute, or all of them if permutation_mode='complete')"""
    
    if permutation_mode=='complete':
        # combine all layer permutations
        combination_list = []
        for layer, perms in layer_permutations.items():
            perms_indices = list(range(len(perms)))
            combination_list.append(perms_indices)
        combinations = np.array(list(itertools.product(*combination_list)))
        # shuffle combinations
        rng,subkey = random.split(rng,2)
        combinations = random.permutation(subkey,combinations)
        combinations = [{layer: x[i] for i,layer in enumerate(layer_permutations.keys())} for x in combinations]
        
    elif permutation_mode=='random':
        # sample one permutation of each layer, randomly
        combinations = []
        for j in range(num_permute):
            combination_list = {}
            for layer, perms in layer_permutations.items():
                rng,subkey = random.split(rng,2)
                perm = random.choice(subkey,np.array(perms))
                combination_list[layer]=perm
            combinations.append(combination_list)
            
    return combinations, rng

def perform_single_permutation(checkpoint_in, permutations):
    """Given model parameters pytree, and a single dictionary of permuted indices per layer,
    return permuted model parameters. Works for convolutional and linear layers. Layers must be named
    accordingly."""
    
    checkpoint = jax.tree_util.tree_map(lambda x: jnp.copy(x), checkpoint_in)
    
    for layer, index_new in permutations.items():
        #index_old = jnp.arange(len(index_new))
        
        next_layer = __get_next_layer(checkpoint, layer)
        if next_layer is None:
            ValueError("Layer name unknown or unavailable (note that you cannot permute the last layer")
        else:
            # permute current
            if "conv" in layer:
                checkpoint[layer]['w'] = checkpoint[layer]['w'][:,:,:,index_new]
                if 'b' in checkpoint[layer]:
                    checkpoint[layer]['b']=checkpoint[layer]['b'][index_new]
            elif "linear" in layer:
                checkpoint[layer]['w']=checkpoint[layer]['w'][:,index_new]
                if 'b' in checkpoint[layer]:
                    checkpoint[layer]['b']=checkpoint[layer]['b'][index_new]
            
            #permute next
            if "conv" in next_layer:
                checkpoint[next_layer]['w']=checkpoint[next_layer]['w'][:,:,index_new,:]
            elif "linear" in next_layer:
                if checkpoint[next_layer]['w'].shape[0] == len(index_new):
                    checkpoint[next_layer]['w']=checkpoint[next_layer]['w'][index_new,:]
                else:
                    #conv flattened and followed by linear
                    new_weights = jnp.copy(checkpoint[next_layer]['w'])
                    block_length = checkpoint[next_layer]['w'].shape[0] // len(index_new)
                    for idx_old, idx_new in enumerate(index_new):
                        for fcdx in range(block_length):
                            offset_old = idx_old * block_length + fcdx
                            offset_new = idx_new * block_length + fcdx
                            new_weights[offset_old,:] = checkpoint[next_layer]['w'][offset_new,:]
                    checkpoint[next_layer]['w'] = new_weights
            
    return checkpoint 

def perform_batch_permutation(checkpoints, permutations):
    """Perform permutations from list of permutation dicts on a list of checkpoints (each is a pytree)
    Returns: a list of pytrees of permuted checkpoints"""
    batch_fun = vmap(perform_single_permutation,in_axes=(0, 0),axis_name='batch')
    stacked_checkpoints = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *checkpoints)
    stacked_permutations = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *permutations)
    batched_result = batch_fun(stacked_checkpoints, stacked_permutations)
    unbatched_results = vmap(lambda x: x, axis_name='batch')(batched_result)
    unbatched_results_list = [
        jax.tree_util.tree_map(lambda *args: dict(zip(checkpoints[i].keys(), args)), *unbatched_results)
        for i in range(len(checkpoints))
    ]
    return unbatched_results_list

def __get_next_layer(checkpoint, layer):
    """Return next layer name if next layer in pytree checkpoint exists."""
    found_current_layer = False
    for layer_name in dict(checkpoint).keys():
        if found_current_layer:
            return layer_name
        if layer_name == layer:
            found_current_layer = True
    return None

if __name__=='__main__':    
    from model_zoo_jax.zoo_dataloader import load_nets
    
    inputs, all_labels = load_nets(n=32, 
                                   data_dir='model_zoo_jax/checkpoints/mnist_smallCNN_fixed_zoo',
                                   flatten=False,
                                   num_checkpoints=1)
    
    params = inputs[0]
    print("param count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print("param tree:", jax.tree_map(lambda x: x.shape, params))
    
    rng = random.PRNGKey(42)
    permutations = permute_checkpoint(rng, params, 
                       permute_layers=["cnn/conv2_d_1","cnn/linear"], 
                       permutation_mode="random",
                       num_permutations=4)
    
    print('-'*50)
    print("permute using for")
    start = time.time()
    result = []
    for i in range(len(inputs)):
        params = inputs[0]
        rng,subkey = random.split(rng)
        result = result + permute_checkpoint(subkey,params,num_permutations=4)
    end = time.time()
    print("one batch takes: {}".format(end-start))
    print("num models: {}".format(len(result)))
    
    print('-'*50)
    print('permute using batch')
    start = time.time()
    rng,subkey = random.split(rng)
    result = permute_batch(subkey,inputs,num_permutations=4)
    end = time.time()
    print("one batch takes: {}".format(end-start))
    print("num models: {}".format(len(result)))
    #print(params["cnn/conv2_d_1"]['w'][0,0,0,:])
    #print(params['cnn/conv2_d_2']['w'][0,0,:,0])
    #print(permutations[0]["cnn/conv2_d_1"]['w'][0,0,0,:])
    #print(permutations[1]["cnn/conv2_d_1"]['w'][0,0,0,:])
    #print(permutations[1]['cnn/conv2_d_2']['w'][0,0,:,0])
    #print(permutations[2]["cnn/conv2_d_1"]['w'][0,0,0,:])
