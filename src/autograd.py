"""A minimal reverse-mode autograd engine over NumPy arrays.

Just enough operators to build and train small neural networks (linear layers,
ReLU/tanh, softmax cross-entropy, self-attention) without a deep-learning
framework. Reverse-mode autodiff: every op records how to push gradients to its
inputs; `backward()` walks the graph in reverse topological order.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)

    # --- core ops -------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            g_self = out.grad @ np.swapaxes(other.data, -1, -2)
            g_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(g_self, self.data.shape)
            other.grad += _unbroadcast(g_other, other.data.shape)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.ones_like(self.data) * g
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,))

        def _backward():
            s = (out.grad * p).sum(axis=axis, keepdims=True)
            self.grad += p * (out.grad - s)
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-12), (self,))

        def _backward():
            self.grad += out.grad / (self.data + 1e-12)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def transpose(self, axes):
        out = Tensor(np.transpose(self.data, axes), (self,))
        inv = np.argsort(axes)

        def _backward():
            self.grad += np.transpose(out.grad, inv)
        out._backward = _backward
        return out

    # --- graph walk -----------------------------------------------------
    def backward(self):
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy. logits: Tensor (N, C); targets: int array."""
    probs = logits.softmax(axis=-1)
    n = logits.data.shape[0]
    onehot = np.zeros_like(logits.data)
    onehot[np.arange(n), targets] = 1.0
    nll = (probs.log() * Tensor(-onehot)).sum() * (1.0 / n)
    return nll
