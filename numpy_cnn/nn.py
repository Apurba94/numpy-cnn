"""Layers, models, optimisers and a training loop."""
import numpy as np

from .tensor import Tensor, concat, conv2d, cross_entropy, global_avg_pool, max_pool, no_grad


class Module:
    training = True

    def parameters(self):
        out = []
        for v in vars(self).values():
            if isinstance(v, Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        out.extend(x.parameters())
        return out

    def state(self):
        """Parameters and running statistics, in a fixed order, for saving."""
        out = [p.data for p in self.parameters()]
        out.extend(m.mean_ for m in self.modules() if isinstance(m, BatchNorm))
        out.extend(m.var_ for m in self.modules() if isinstance(m, BatchNorm))
        return out

    def modules(self):
        yield self
        for v in vars(self).values():
            if isinstance(v, Module):
                yield from v.modules()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        yield from x.modules()

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *a):
        return self.forward(*a)

    def count(self):
        return sum(p.data.size for p in self.parameters())

    def save(self, path):
        np.savez(path, *self.state())

    def load(self, path):
        data = np.load(path)
        for i, target in enumerate(self.state()):
            saved = data[f"arr_{i}"]
            if saved.shape != target.shape:
                raise ValueError(f"saved array {i} has shape {saved.shape}, the model wants {target.shape}")
            target[...] = saved


class Linear(Module):
    def __init__(self, n_in, n_out, rng):
        self.w = Tensor(rng.normal(0, np.sqrt(2 / n_in), (n_out, n_in)), requires_grad=True)
        self.b = Tensor(np.zeros(n_out), requires_grad=True)

    def forward(self, x):
        return x @ self.w.transpose(1, 0) + self.b


class Conv(Module):
    def __init__(self, c_in, c_out, k, rng, stride=1, padding=0):
        self.stride, self.padding = stride, padding
        std = np.sqrt(2 / (c_in * k * k))                 # He initialisation
        self.w = Tensor(rng.normal(0, std, (c_out, c_in, k, k)), requires_grad=True)
        self.b = Tensor(np.zeros(c_out), requires_grad=True)

    def forward(self, x):
        return conv2d(x, self.w, self.b, self.stride, self.padding)


class BatchNorm(Module):
    """Batch normalisation over channels, for N×C×H×W or N×C inputs."""

    def __init__(self, c, momentum=0.1):
        self.g = Tensor(np.ones(c), requires_grad=True)
        self.b = Tensor(np.zeros(c), requires_grad=True)
        self.mean_, self.var_ = np.zeros(c), np.ones(c)
        self.momentum = momentum

    def forward(self, x):
        axes = (0, 2, 3) if len(x.shape) == 4 else (0,)
        shape = (1, -1, 1, 1) if len(x.shape) == 4 else (1, -1)
        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            self.mean_ = (1 - self.momentum) * self.mean_ + self.momentum * mean.data.reshape(-1)
            self.var_ = (1 - self.momentum) * self.var_ + self.momentum * var.data.reshape(-1)
        else:
            mean, var = Tensor(self.mean_.reshape(shape)), Tensor(self.var_.reshape(shape))
        z = (x - mean) / ((var + 1e-5) ** 0.5)
        return z * self.g.reshape(*shape) + self.b.reshape(*shape)


class Dropout(Module):
    def __init__(self, rate, rng):
        self.rate, self.rng = rate, rng

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        keep = (self.rng.random(x.shape) >= self.rate) / (1 - self.rate)
        return x * keep


class Residual(Module):
    """Two 3×3 convolutions with a skip connection, as in ResNet."""

    def __init__(self, c, rng):
        self.a = Conv(c, c, 3, rng, padding=1)
        self.na = BatchNorm(c)
        self.b = Conv(c, c, 3, rng, padding=1)
        self.nb = BatchNorm(c)

    def forward(self, x):
        y = self.na(self.a(x)).relu()
        return (self.nb(self.b(y)) + x).relu()


class SmallResNet(Module):
    def __init__(self, classes, channels=1, width=8, blocks=1, seed=0):
        rng = np.random.default_rng(seed)
        self.first = Conv(channels, width, 3, rng, padding=1)
        self.norm = BatchNorm(width)
        self.blocks = [Residual(width, rng) for _ in range(blocks)]
        self.down = Conv(width, width * 2, 3, rng, stride=2, padding=1)
        self.norm2 = BatchNorm(width * 2)
        self.head = Linear(width * 2, classes, rng)

    def forward(self, x):
        x = max_pool(self.norm(self.first(x)).relu())
        for b in self.blocks:
            x = b(x)
        x = self.norm2(self.down(x)).relu()
        return self.head(global_avg_pool(x))


class SGD:
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0):
        self.params, self.lr, self.m, self.wd = params, lr, momentum, weight_decay
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            v *= self.m
            v -= self.lr * g
            p.data += v

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = params, lr, betas[0], betas[1], eps, weight_decay
        self.m1 = [np.zeros_like(p.data) for p in params]
        self.m2 = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, a, b in zip(self.params, self.m1, self.m2):
            if p.grad is None:
                continue
            a *= self.b1
            a += (1 - self.b1) * p.grad
            b *= self.b2
            b += (1 - self.b2) * p.grad ** 2
            ah = a / (1 - self.b1 ** self.t)
            bh = b / (1 - self.b2 ** self.t)
            p.data -= self.lr * ah / (np.sqrt(bh) + self.eps) + self.lr * self.wd * p.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def cosine_lr(base, step, total, warmup=0):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + np.cos(np.pi * min(t, 1)))


def predict(model, x, batch=128):
    model.eval()
    out = []
    with no_grad():
        for i in range(0, len(x), batch):
            out.append(model(Tensor(x[i:i + batch])).data)
    return np.concatenate(out)


def fit(model, x, y, xval=None, yval=None, epochs=5, batch=32, lr=0.001, seed=0, optimiser=None, log=None):
    """Train with shuffled mini-batches and a cosine schedule. Returns a history of losses and accuracy."""
    rng = np.random.default_rng(seed)
    opt = optimiser or Adam(model.parameters(), lr=lr)
    base = opt.lr
    steps = epochs * max(1, len(x) // batch)
    step = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x))
        total, seen = 0.0, 0
        for i in range(0, len(x) - batch + 1, batch):
            idx = order[i:i + batch]
            opt.lr = cosine_lr(base, step, steps)
            loss = cross_entropy(model(Tensor(x[idx])), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
            step += 1
        row = {"epoch": epoch + 1, "loss": total / max(seen, 1)}
        if xval is not None:
            row["val_accuracy"] = float((predict(model, xval).argmax(axis=1) == yval).mean())
        history.append(row)
        if log:
            log(row)
    return history


def shapes(n, size=16, classes=3, seed=0, noise=0.15):
    """A small synthetic image set: bars, crosses and rings at random positions."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 1, size, size))
    y = rng.integers(0, classes, n)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        cy, cx = rng.integers(5, size - 5, 2)
        img = np.zeros((size, size))
        r = rng.integers(3, 5)
        if y[i] == 0:
            img[cy - 1:cy + 2, cx - r:cx + r + 1] = 1
        elif y[i] == 1:
            img[cy, cx - r:cx + r + 1] = 1
            img[cy - r:cy + r + 1, cx] = 1
        else:
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            img[(d > r - 1.2) & (d < r + 0.2)] = 1
        x[i, 0] = img + rng.normal(0, noise, img.shape)
    return x, y
