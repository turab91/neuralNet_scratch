"""
Feed inputs into the network in batches.
"""
from typing import Iterator, NamedTuple
import numpy as np
from turabnet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])

class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        assert inputs.shape[0] == targets.shape[0], \
            f"inputs and targets shape should be same. Got {inputs.shape[0]} != {targets.shape[0]}"

        starts = np.arange(0, len(inputs), self.batch_size)
        # shuffle the batch
        if self.shuffle:
            np.random.shuffle(starts)
        # TODO - one batch might have too few inputs. rather distribute them among all batch.
        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)

