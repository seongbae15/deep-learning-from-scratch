from dataset.mnist import load_mnist


def train():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)


def main():
    train()


if __name__ == "__main__":
    main()
