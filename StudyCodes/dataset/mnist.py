import os

key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    for v in key_file.values():
        _download(v)


def init_mnist():
    download_mnist()


def load_mnist(normalize=True, flatten=True, one_hot_lable=False):
    if not os.path.exists(save_file):
        init_mnist()
