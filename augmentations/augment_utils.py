import jax
import jax.numpy as jnp
from augmentations.permutation_augmentation import permute_checkpoint,permute_batch
#from permutation_augmentation import permute_checkpoint,permute_batch

def augment_batch(rng, data, labels, num_p=3, keep_original=True,
                  layers = ["cnn/conv2_d","cnn/conv2_d_1","cnn/conv2_d_2", "cnn/linear"]):
    data_new = permute_batch(rng, data, num_permutations=num_p,
                            permute_layers=layers,
                            keep_original=keep_original)
    #print(data_new[0])
    num_labels = num_p+1 if keep_original else num_p
    labels_new = [labels[j] for i in range(num_labels) for j in range(len(labels))]
    return data_new, jnp.array(labels_new)

def augment(rng,data, labels,num_p=4,verbose=True,keep_original=True,
            layers= ["cnn/conv2_d","cnn/conv2_d_1","cnn/conv2_d_2", "cnn/linear"]):
    data_new = []
    labels_new = []
    i=0
    for datapoint,label in zip(data,labels):
        rng,subkey = jax.random.split(rng)
        permuted = permute_checkpoint(subkey,datapoint,num_permutations=num_p,
                                      permute_layers=layers,
                                      keep_original=keep_original)
        data_new = data_new + permuted
        if keep_original:
            labels_new = labels_new + [label for i in range(num_p+1)]
        else:
            labels_new = labels_new + [label for i in range(num_p)]
        if verbose:
            if i%100==0:
                print('Augmented: {}/{}'.format(i,len(labels)))
            i = i+1
    return data_new,jnp.array(labels_new)

if __name__ == '__main__':
    from model_zoo_jax.zoo_dataloader import load_nets,shuffle_data
    from jax import random
    
    rng = random.PRNGKey(42)
    
    inputs, all_labels = load_nets(n=16, 
                                   data_dir='model_zoo_jax/checkpoints/mnist_smallCNN_fixed_zoo',
                                   flatten=False,
                                   num_checkpoints=1)
    labels = all_labels["class_dropped"]
    
    rng, subkey = random.split(rng)
    filtered_inputs, filtered_labels = shuffle_data(subkey,inputs,labels,chunks=1)
    filtered_inputs, filtered_labels = inputs,labels
    
    rng, subkey = random.split(rng)
    images,labels = augment_batch(subkey,filtered_inputs,filtered_labels,num_p=1,keep_original=False)
    print(images[0]['cnn/conv2_d']['w'].shape)
    
    batch = jax.tree_util.tree_map(lambda *x: list(x), *images)
    print(batch['cnn/conv2_d']['w'][0].shape)
    