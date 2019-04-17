"""
The neural net is made up of layers.
Each layer  pass its inputs forward and propagate gradients backward.

example:
a nueral net might look like
inputs -> Linear -> Tanh -> Linear -> output

"""
from typing import Dict
import numpy as np
from turabnet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        maps inputs to outputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        r"""
        Backpropagate  gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size < 0 or not isinstance(input_size, int):
            raise TypeError("input_size should be a positive integer")
        if output_size < 0 or not isinstance(output_size, int):
            raise TypeError("output_size should be a positive integer")
        # TODO use better initialization
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        inputs : (batch_size, input_size)
        outputs: (batch_size, output_size)
        outputs = input @ w + b
        """
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs should be a Tensor")
        assert inputs.ndim == 2, \
            f"inputs should be of shape (batch_size, input_size). Got {inputs.shape}"
        assert inputs.shape[1] == self.params["w"].shape[0], \
            f"inputs has unexpected input_size. got {inputs.shape[1]} expected {self.params['w'].shape[0]}"

        self.inputs = inputs
        return np.dot(inputs, self.params["w"]) + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        r"""
        if y = f(x) and x = a * b + c
        then:
        dy / da = f'(x) * b
        dy / db = f'(x) * a
        dy / dc = f'(x)
        """
        if not isinstance(grad, Tensor):
            raise TypeError("grad should be a Tensor")
        assert grad.shape[1] == self.params["w"].shape[1], \
            f"grad has unexpected output_size. got {grad.shape[1]} expected {self.params['w'].shape[1]}"

        self.grads["b"] = np.sum(grad, axis=0)                # sum over batch
        self.grads["w"] = np.dot(self.inputs.T, grad)

        assert self.params["w"].shape == self.grads["w"].shape, f" shape w != dw. Got {self.params['w'].shape} , {self.grads['w']}"
        assert self.params["b"].shape == self.grads["b"].shape, f" shape b != db. Got {self.params['b'].shape} , {self.grads['b']}"

        return np.dot(grad, self.params["w"].T)


class Tanh(Layer):
    r"""
    Squeezes inputs value between -1 and 1. applies element-wise operation
    """
    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs should be a Tensor")
        assert inputs.ndim == 2, \
            f"inputs should be of shape (batch_size, input_size). Got {inputs.shape}"

        self.y = np.tanh(inputs)
        return self.y

    def backward(self, grad: Tensor) -> Tensor:
        r"""
        if y = f(x) and x = g(z)
        then,
        dy / dz = f'(x) * g'(z)
        """
        if not isinstance(grad, Tensor):
            raise TypeError("grad should be a Tensor")

        return grad * (1 - self.y ** 2)


class Sigmoid(Layer):
    r"""
    Squeezes inputs value between 0 and 1. applies element-wise operation
    """
    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor):
            raise TypeError("inputs should be a Tensor")
        assert inputs.ndim == 2, \
            f"inputs should be of shape (batch_size, input_size). Got {inputs.shape}"

        self.y = np.where(inputs > 0,
                          1.0 / (1.0 + np.exp(-inputs)),
                          np.exp(inputs) / (1.0 + np.exp(inputs)))
        return self.y

    def backward(self, grad: Tensor):
        """
        if y = f(x) and x = g(z)
        then,
        dy / dz = f'(x) * g'(z)
        """
        if not isinstance(grad, Tensor):
            raise TypeError("grad should be a Tensor")

        return grad * self.y * (1 - self.y)
