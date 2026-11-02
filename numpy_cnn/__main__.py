"""Train a small ResNet on synthetic shapes: python -m numpy_cnn [epochs]"""
import sys

from . import SmallResNet, fit, shapes


def main(argv):
    epochs = int(argv[1]) if len(argv) > 1 else 6
    x, y = shapes(1200, seed=0)
    xv, yv = shapes(300, seed=1)
    model = SmallResNet(3)
    print(f"{model.count()} parameters")
    fit(model, x, y, xv, yv, epochs=epochs, lr=0.01, log=lambda r: print(f"epoch {r['epoch']}: loss {r['loss']:.4f}, validation accuracy {r['val_accuracy']:.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
