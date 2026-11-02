# NumPy CNN

A deep-learning framework written from scratch in NumPy: automatic differentiation, convolutions, batch norm, residual blocks, optimisers and a training loop. It is small enough to read in an afternoon, and it does what PyTorch does at its core.

```python
from numpy_cnn import SmallResNet, fit, shapes

x, y = shapes(1200, seed=0)
x_val, y_val = shapes(300, seed=1)

model = SmallResNet(classes=3)
fit(model, x, y, x_val, y_val, epochs=6, lr=0.01, log=print)
```

Or run the demo:

```
python -m numpy_cnn 6
```

## What is inside

| Part | File | Contents |
|---|---|---|
| Autograd | `tensor.py` | A `Tensor` that records operations, and `backward()` in reverse topological order, with broadcasting handled properly |
| Vision | `tensor.py` | `conv2d` with stride and padding, max pooling, global average pooling, and stable cross-entropy |
| Layers | `nn.py` | `Linear`, `Conv`, `BatchNorm`, `Dropout`, `Residual`, `SmallResNet` |
| Training | `nn.py` | `SGD` with momentum, `Adam`, cosine learning rate with warm-up, `fit`, `predict`, save and load |

## How it is tested

The backward rule of every operation is compared with a central-difference numerical gradient, including `conv2d` at several strides and paddings, pooling, broadcasting, shared nodes and a small network from end to end. The convolution is also checked against a direct sum. Then a small ResNet must reach over 90% on held-out synthetic shapes, and the same seed must give identical weights.

## Limits

It runs on the CPU in double precision and is slow: expect seconds per epoch on tiny images, and use PyTorch for real work. Weight initialisation and layer choices are simple. The shapes dataset is synthetic, so the accuracy here shows that the framework learns, and says nothing about real images.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.
