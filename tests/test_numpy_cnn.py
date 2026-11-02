import os
import tempfile
import unittest

import numpy as np

from numpy_cnn import SGD, Adam, BatchNorm, Conv, Dropout, Linear, Module, Residual, SmallResNet, Tensor, concat, conv2d, cosine_lr, cross_entropy, fit, global_avg_pool, max_pool, no_grad, predict, shapes


def numeric(f, x, eps=1e-6):
    """Central-difference gradient: the independent check on every backward rule."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        a = f()
        x[i] = old - eps
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
    return g


class GradCheck(unittest.TestCase):
    def check(self, fn, *arrays, delta=1e-5):
        ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
        out = fn(*ts)
        out.backward()
        for t in ts:
            want = numeric(lambda: fn(*[Tensor(x.data) for x in ts]).item(), t.data)
            self.assertTrue(np.allclose(t.grad, want, atol=delta, rtol=1e-4), f"\n{t.grad}\n{want}")

    def r(self, *shape, seed=0):
        return np.random.default_rng(seed).normal(size=shape)

    def test_arithmetic(self):
        self.check(lambda a, b: ((a * b + a - b) / (b * b + 1)).sum(), self.r(3, 2), self.r(3, 2, seed=1))

    def test_broadcasting(self):
        self.check(lambda a, b: (a * b + a).sum(), self.r(3, 1), self.r(1, 4, seed=1))
        self.check(lambda a, b: (a + b).sum(), self.r(2, 3), self.r(3, seed=1))

    def test_powers_and_functions(self):
        p = np.abs(self.r(4)) + 0.5
        self.check(lambda a: (a ** 3).sum() + a.exp().sum() + a.log().sum() + a.tanh().sum() + a.sigmoid().sum(), p)

    def test_relu(self):
        self.check(lambda a: a.relu().sum(), self.r(5) + 0.1)

    def test_matmul(self):
        self.check(lambda a, b: (a @ b).sum(), self.r(3, 4), self.r(4, 2, seed=1))
        self.check(lambda a, b: (a @ b).sum(), self.r(3, 4), self.r(4, seed=1))
        self.check(lambda a, b: a @ b, self.r(4), self.r(4, seed=1))

    def test_shapes(self):
        self.check(lambda a: (a.reshape(6) * Tensor(np.arange(6.0))).sum(), self.r(2, 3))
        self.check(lambda a: (a.transpose(1, 0) ** 2).sum(), self.r(2, 3))
        self.check(lambda a: (a.sum(axis=1) ** 2).sum() + (a.mean(axis=0, keepdims=True) ** 2).sum(), self.r(2, 3))
        self.check(lambda a: (a[1:, 0] ** 2).sum(), self.r(3, 2))

    def test_concat(self):
        self.check(lambda a, b: (concat([a, b], 1) ** 2).sum(), self.r(2, 2), self.r(2, 3, seed=1))

    def test_shared_node(self):
        self.check(lambda a: (a * a + a).sum(), self.r(3))

    def test_convolution(self):
        for stride, padding in ((1, 0), (1, 1), (2, 1), (2, 0)):
            self.check(lambda x, w, b: (conv2d(x, w, b, stride, padding) ** 2).sum(), self.r(2, 2, 6, 6), self.r(3, 2, 3, 3, seed=1), self.r(3, seed=2))

    def test_max_pool(self):
        self.check(lambda x: (max_pool(x) ** 2).sum(), self.r(2, 2, 6, 6))
        self.check(lambda x: (max_pool(x, 3, 2) ** 2).sum(), self.r(1, 1, 7, 7))

    def test_global_pool(self):
        self.check(lambda x: (global_avg_pool(x) ** 2).sum(), self.r(2, 2, 3, 3))

    def test_cross_entropy(self):
        self.check(lambda a: cross_entropy(a, [0, 2, 1]), self.r(3, 4))

    def test_whole_network(self):
        rng = np.random.default_rng(3)
        net = SmallResNet(3, width=2, seed=1)
        net.train()
        x = rng.normal(size=(4, 1, 8, 8))
        y = [0, 1, 2, 1]
        loss = cross_entropy(net(Tensor(x)), y)
        loss.backward()
        for p in net.parameters()[:4] + net.parameters()[-2:]:
            def f():
                net_copy = cross_entropy(net(Tensor(x)), y)
                return net_copy.item()
            near = numeric(f, p.data) if p.data.size < 40 else None
            if near is not None:
                self.assertTrue(np.allclose(p.grad, near, atol=2e-4, rtol=2e-2))

    def test_conv_matches_a_direct_sum(self):
        x = self.r(1, 2, 5, 5)
        w = self.r(2, 2, 3, 3, seed=1)
        out = conv2d(Tensor(x), Tensor(w), stride=1, padding=0).data
        want = np.zeros(out.shape)
        for o in range(2):
            for yy in range(3):
                for xx in range(3):
                    want[0, o, yy, xx] = (x[0, :, yy:yy + 3, xx:xx + 3] * w[o]).sum()
        self.assertTrue(np.allclose(out, want))


class TestLayers(unittest.TestCase):
    def test_shapes(self):
        rng = np.random.default_rng(0)
        self.assertEqual(Conv(3, 8, 3, rng, stride=2, padding=1)(Tensor(np.zeros((2, 3, 16, 16)))).shape, (2, 8, 8, 8))
        self.assertEqual(Linear(5, 2, rng)(Tensor(np.zeros((4, 5)))).shape, (4, 2))
        self.assertEqual(Residual(4, rng)(Tensor(np.zeros((1, 4, 8, 8)))).shape, (1, 4, 8, 8))
        self.assertEqual(max_pool(Tensor(np.zeros((1, 1, 8, 8)))).shape, (1, 1, 4, 4))

    def test_batch_norm_training_output(self):
        b = BatchNorm(3)
        x = np.random.default_rng(0).normal(5, 3, (64, 3, 4, 4))
        y = b(Tensor(x)).data
        self.assertTrue(np.allclose(y.mean(axis=(0, 2, 3)), 0, atol=1e-6))
        self.assertTrue(np.allclose(y.std(axis=(0, 2, 3)), 1, atol=1e-2))

    def test_batch_norm_eval_uses_running_stats(self):
        b = BatchNorm(1, momentum=1.0)
        b(Tensor(np.array([[1.0], [3.0]])))
        b.eval()
        self.assertAlmostEqual(float(b(Tensor(np.array([[2.0]]))).data[0, 0]), 0, places=6)

    def test_dropout(self):
        d = Dropout(0.5, np.random.default_rng(0))
        x = Tensor(np.ones(10000))
        y = d(x).data
        self.assertAlmostEqual(y.mean(), 1, delta=0.05)
        self.assertAlmostEqual((y == 0).mean(), 0.5, delta=0.03)
        d.eval()
        self.assertTrue(np.array_equal(d(x).data, np.ones(10000)))

    def test_no_grad_builds_no_graph(self):
        a = Tensor(np.ones(2), requires_grad=True)
        with no_grad():
            b = a * 2
        self.assertEqual(b.parents, ())
        self.assertFalse(b.requires_grad)

    def test_parameters_found_through_lists(self):
        self.assertEqual(len(SmallResNet(2, blocks=2).parameters()), 10 + 2 * 8)

    def test_cross_entropy_is_stable(self):
        v = cross_entropy(Tensor(np.array([[1000.0, -1000.0]])), [0]).item()
        self.assertAlmostEqual(v, 0)

    def test_save_and_load(self):
        a, b = SmallResNet(3, width=2, seed=1), SmallResNet(3, width=2, seed=2)
        a.train()
        a(Tensor(np.random.default_rng(0).normal(size=(4, 1, 8, 8))))
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.npz")
        a.save(p)
        b.load(p)
        x = np.random.default_rng(1).normal(size=(2, 1, 8, 8))
        self.assertTrue(np.allclose(predict(a, x), predict(b, x)))
        c = SmallResNet(3, width=4)
        with self.assertRaises(ValueError):
            c.load(p)


class TestTraining(unittest.TestCase):
    def test_optimisers_minimise(self):
        for make in (lambda p: SGD(p, 0.1, 0.9), lambda p: Adam(p, 0.1)):
            w = Tensor(np.array([5.0, -3.0]), requires_grad=True)
            opt = make([w])
            for _ in range(300):
                loss = (w * w).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertLess(np.abs(w.data).max(), 0.05)

    def test_adam_first_step_is_about_lr(self):
        w = Tensor(np.array([1.0]), requires_grad=True)
        opt = Adam([w], 0.01)
        (w * 100.0).sum().backward()
        opt.step()
        self.assertAlmostEqual(w.data[0], 0.99, places=6)

    def test_schedule(self):
        self.assertAlmostEqual(cosine_lr(1, 0, 100), 1)
        self.assertAlmostEqual(cosine_lr(1, 100, 100), 0)
        self.assertAlmostEqual(cosine_lr(1, 50, 100), 0.5)
        self.assertAlmostEqual(cosine_lr(1, 4, 100, warmup=10), 0.5)

    def test_linear_model_learns_a_rule(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 4))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)

        class Net(Module):
            def __init__(self):
                self.a = Linear(4, 2, np.random.default_rng(1))

            def forward(self, t):
                return self.a(t)
        n = Net()
        fit(n, x, y, epochs=10, lr=0.05)
        self.assertGreater((predict(n, x).argmax(axis=1) == y).mean(), 0.97)

    def test_small_resnet_learns_shapes(self):
        x, y = shapes(500, seed=0)
        xv, yv = shapes(200, seed=1)
        m = SmallResNet(3, width=6)
        h = fit(m, x, y, xv, yv, epochs=4, lr=0.01)
        self.assertLess(h[-1]["loss"], h[0]["loss"])
        self.assertGreater(h[-1]["val_accuracy"], 0.9)

    def test_same_seed_same_model(self):
        x, y = shapes(64, seed=0)
        runs = []
        for _ in range(2):
            m = SmallResNet(3, width=2, seed=5)
            fit(m, x, y, epochs=1, batch=16, seed=3)
            runs.append(m.parameters()[0].data.copy())
        self.assertTrue(np.array_equal(runs[0], runs[1]))


if __name__ == "__main__":
    unittest.main()
