from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet


def train():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    network = DeepConvNet()


def main():
    train()


if __name__ == "__main__":
    main()
