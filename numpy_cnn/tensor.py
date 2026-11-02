"""Reverse-mode automatic differentiation on NumPy arrays.

Each operation records its inputs and a function that passes gradients back to them.
`backward()` walks that graph in reverse topological order.
"""
import numpy as np


def _unbroadcast(grad, shape):
    """Undo NumPy broadcasting, so a gradient has the shape of its own tensor."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __array_priority__ = 100
    grad_enabled = True

    def __init__(self, data, parents=(), back=None, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.parents = parents
        self.back = back
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    def item(self):
        return float(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def zero_grad(self):
        self.grad = None

    def backward(self, seed=None):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node.parents if p.requires_grad)
        self.grad = np.ones_like(self.data) if seed is None else seed
        for node in reversed(order):
            if node.back is not None and node.grad is not None:
                for parent, g in node.back(node.grad):
                    if parent.requires_grad:
                        g = _unbroadcast(g, parent.shape)
                        parent.grad = g if parent.grad is None else parent.grad + g

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    def _op(self, data, parents, back):
        if not Tensor.grad_enabled:
            return Tensor(data)
        return Tensor(data, parents, back)

    def __add__(self, o):
        o = Tensor.wrap(o)
        return self._op(self.data + o.data, (self, o), lambda g: [(self, g), (o, g)])

    __radd__ = __add__

    def __neg__(self):
        return self._op(-self.data, (self,), lambda g: [(self, -g)])

    def __sub__(self, o):
        return self + (-Tensor.wrap(o))

    def __rsub__(self, o):
        return Tensor.wrap(o) + (-self)

    def __mul__(self, o):
        o = Tensor.wrap(o)
        return self._op(self.data * o.data, (self, o), lambda g: [(self, g * o.data), (o, g * self.data)])

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Tensor.wrap(o)
        return self._op(self.data / o.data, (self, o), lambda g: [(self, g / o.data), (o, -g * self.data / o.data ** 2)])

    def __pow__(self, p):
        return self._op(self.data ** p, (self,), lambda g: [(self, g * p * self.data ** (p - 1))])

    def __matmul__(self, o):
        o = Tensor.wrap(o)

        def back(g):
            a, b = self.data, o.data
            if a.ndim == 1 and b.ndim == 1:
                return [(self, g * b), (o, g * a)]
            if b.ndim == 1:
                return [(self, np.expand_dims(g, -1) * b), (o, a.T @ g)]
            return [(self, g @ np.swapaxes(b, -1, -2)), (o, np.swapaxes(a, -1, -2) @ g)]
        return self._op(self.data @ o.data, (self, o), back)

    def sum(self, axis=None, keepdims=False):
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return [(self, np.broadcast_to(g, self.shape))]
        return self._op(self.data.sum(axis=axis, keepdims=keepdims), (self,), back)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis, keepdims) / float(n)

    def reshape(self, *shape):
        return self._op(self.data.reshape(*shape), (self,), lambda g: [(self, g.reshape(self.shape))])

    def transpose(self, *axes):
        inverse = np.argsort(axes)
        return self._op(self.data.transpose(*axes), (self,), lambda g: [(self, g.transpose(*inverse))])

    def __getitem__(self, key):
        def back(g):
            out = np.zeros_like(self.data)
            np.add.at(out, key, g)
            return [(self, out)]
        return self._op(self.data[key], (self,), back)

    def exp(self):
        return self._op(np.exp(self.data), (self,), lambda g: [(self, g * np.exp(self.data))])

    def log(self):
        return self._op(np.log(self.data), (self,), lambda g: [(self, g / self.data)])

    def relu(self):
        return self._op(np.maximum(self.data, 0), (self,), lambda g: [(self, g * (self.data > 0))])

    def tanh(self):
        return self._op(np.tanh(self.data), (self,), lambda g: [(self, g * (1 - np.tanh(self.data) ** 2))])

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        return self._op(s, (self,), lambda g: [(self, g * s * (1 - s))])


def concat(items, axis=0):
    sizes = [t.shape[axis] for t in items]
    cuts = np.cumsum(sizes)[:-1]

    def back(g):
        return list(zip(items, np.split(g, cuts, axis=axis)))
    return items[0]._op(np.concatenate([t.data for t in items], axis=axis), tuple(items), back)


class no_grad:
    """Turn off graph building, for inference."""

    def __enter__(self):
        self.old = Tensor.grad_enabled
        Tensor.grad_enabled = False

    def __exit__(self, *exc):
        Tensor.grad_enabled = self.old


def _windows(x, k, s, p):
    """x: N×C×H×W. Returns a view N×C×H'×W'×k×k of the sliding windows, after zero padding."""
    if p:
        x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    v = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))
    return v[:, :, ::s, ::s]


def conv2d(x, w, b=None, stride=1, padding=0):
    """x: N×C×H×W. w: Cout×Cin×k×k. Gradients come from the same windows used going forward."""
    k = w.shape[2]
    win = _windows(x.data, k, stride, padding)
    out = np.einsum("ncyxij,ocij->noyx", win, w.data, optimize=True)
    H, W = out.shape[2:]

    def back(g):
        grad_w = np.einsum("noyx,ncyxij->ocij", g, win, optimize=True)
        n, c = x.shape[:2]
        h = x.shape[2] + 2 * padding
        wd = x.shape[3] + 2 * padding
        grad_x = np.zeros((n, c, h, wd))
        per = np.einsum("noyx,ocij->ncyxij", g, w.data, optimize=True)
        for y in range(H):
            for z in range(W):
                grad_x[:, :, y * stride:y * stride + k, z * stride:z * stride + k] += per[:, :, y, z]
        if padding:
            grad_x = grad_x[:, :, padding:-padding, padding:-padding]
        return [(x, grad_x), (w, grad_w)]
    y = x._op(out, (x, w), back)
    return y if b is None else y + b.reshape(1, -1, 1, 1)


def max_pool(x, k=2, stride=None):
    stride = stride or k
    win = _windows(x.data, k, stride, 0)
    n, c, H, W = win.shape[:4]
    flat = win.reshape(n, c, H, W, -1)
    which = flat.argmax(axis=-1)
    out = flat.max(axis=-1)

    def back(g):
        grad = np.zeros_like(x.data)
        nn, cc, yy, zz = np.indices(which.shape, sparse=False)
        np.add.at(grad, (nn, cc, yy * stride + which // k, zz * stride + which % k), g)
        return [(x, grad)]
    return x._op(out, (x,), back)


def global_avg_pool(x):
    return x.mean(axis=(2, 3))


def cross_entropy(logits, labels):
    """Mean softmax cross-entropy, computed stably from logits."""
    labels = np.asarray(labels, int)
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    n = len(labels)
    p = np.exp(log_p)

    def back(g):
        grad = p.copy()
        grad[np.arange(n), labels] -= 1
        return [(logits, grad * g / n)]
    return logits._op(-log_p[np.arange(n), labels].mean(), (logits,), back)
