from jax.typing import ArrayLike
from typing import Tuple, Iterator


def data_iterator(
        inputs: ArrayLike, 
        targets: ArrayLike, 
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):  # TODO: no warning thrown if last batches are differently sized
            break
        yield dict(input=inputs[i:i + batchsize], 
                   target=targets[i:i + batchsize])


# TODO replace all this with huggingface datasets
def split_data(data: ArrayLike, targets: ArrayLike, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return (data[:split_index], targets[:split_index], 
            data[split_index:], targets[split_index:])