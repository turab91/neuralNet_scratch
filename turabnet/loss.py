"""
A loss function measures how good our predictions are.
we can use this to adjust the parameters of our network
"""
import numpy as np
from turabnet.tensor import Tensor


class Loss:

    def loss(self, y_pred: Tensor, y_true: Tensor) -> float:
        r"""
         loss is effected by prediction
        """
        raise NotImplementedError

    def grad(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        r"""
        partial derivative of loss w.r.t each of the prediction.
        i.e. how each prediction effects the loss
        """
        raise NotImplementedError


class MSE(Loss):

    def loss(self, y_pred: Tensor, y_true: Tensor) -> float:
        r"""
        y_pred: (batch_size, num_classes) prob.dist
        y_true: (batch_size, num_classes) one-hot

        Returns
        -------
            loss: average batch loss
        """
        assert y_pred.shape == y_true.shape, \
            f"shape of y_pred and y_true should be same. Got {y_pred.shape} != {y_true.shape}"

        return np.mean((y_pred - y_true) ** 2)

    def grad(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert y_pred.shape == y_true.shape, \
            f"shape of y_pred and y_true should be same. Got {y_pred.shape} != {y_true.shape}"

        return 2 * (y_pred - y_true)

def Softmax(inputs: Tensor) -> Tensor:
    r"""
    stable softmax
    """
    a = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    a = a / np.sum(a, axis=1, keepdims=True)
    assert inputs.shape == a.shape
    return a


class Softmax_cross_entropy_with_logits(Loss):

    def loss(self, y_pred: Tensor, y_true: Tensor) -> float:
        """
        y_pred: (batch_size, num_classes)
        y_true: (batch_size)              labels

        Returns
        -------
            loss: average batch loss
        """
        assert y_true.ndim == 1, f"y_true should be of shape (batch_size). Got {y_true.shape}"
        assert y_pred.shape[0] == y_true.shape[0], \
            f"batch_size of y_pred and y_true should be same. Got {y_pred.shape[0]} != {y_true.shape[0]}"

        m = y_true.shape[0]
        self.prob = Softmax(y_pred)
        log_likelihood = -np.log(self.prob[range(m), y_true])
        loss = np.mean(log_likelihood)
        return loss

    def grad(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        m = y_true.shape[0]
        grad = self.prob
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad

