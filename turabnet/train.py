"""
a function to train a neural net
"""

from turabnet.tensor import Tensor
from turabnet.nn import NeuralNet
from turabnet.loss import Loss, MSE
from turabnet.optim import Optimizer, SGD
from turabnet.data import DataIterator, BatchIterator
from typing import List

def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 100,
          iterator: DataIterator = BatchIterator(batch_size=32, shuffle=True),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(lr=0.01),
          verbose=True, print_every=10) -> List[float]:

    all_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            # Forward prop
            predicted = net.forward(batch.inputs)
            # Loss
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            # Back prop
            net.backward(grad)
            # update
            optimizer.step(net)
        all_loss.append(epoch_loss)

        if verbose and epoch == 0:
            print(f"epoch: {epoch + 1 :3d} loss: {epoch_loss :.3f}")
        if verbose and (epoch + 1) % print_every == 0:
            print(f"epoch: {epoch + 1 :3d} loss: {epoch_loss :.3f}")

    return all_loss

