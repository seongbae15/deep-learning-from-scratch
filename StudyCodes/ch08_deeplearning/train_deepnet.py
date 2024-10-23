import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
print(sys.path)

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet


def train():
    x_train, t_train, x_test, t_test = load_mnist(flatten=False)
    network = DeepConvNet()
    print(network)


def main():
    train()


if __name__ == "__main__":
    main()
