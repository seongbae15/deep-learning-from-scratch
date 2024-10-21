import os
import gzip
import pickle

try:
    import urllib.request
except ImportError:
    raise ImportError("You should use python 3.x")

import numpy as np

KEY_FILE = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = DATASET_DIR + "/mnist.pkl"
URL_BASE = "http://yann.lecun.com/exdb/mnist/"
IMG_SIZE = 784


def _download(file_name):
    file_path = DATASET_DIR + "/" + file_name
    if os.path.exists(file_path):
        return
    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(URL_BASE + file_name, file_path)
    print("Done")


def download_mnist():
    for v in KEY_FILE.values():
        _download(v)


def _load_img(file_name):
    file_path = DATASET_DIR + "/" + file_name
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, IMG_SIZE)
    print("Done")
    return data


def _load_label(file_name):
    file_path = DATASET_DIR + "/" + file_name
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=16)
    print("Done")
    return labels


def _convert_numpy():
    dataset = {}
    dataset["train_img"] = _load_img(KEY_FILE["train_img"])
    dataset["train_label"] = _load_label(KEY_FILE["train_label"])
    dataset["test_img"] = _load_img(KEY_FILE["test_img"])
    dataset["test_label"] = _load_label(KEY_FILE["test_label"])
    return dataset


def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file...")
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T


def load_mnist(normalize=True, flatten=True, one_hot_lable=False):
    if not os.path.exists(SAVE_FILE):
        init_mnist()

    with open(SAVE_FILE, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_lable:
        for key in ("train_label", "test_label"):
            dataset[key] = _change_one_hot_label(dataset[key])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    return (
        dataset["train_img"],
        dataset["train_label"],
        dataset["test_img"],
        dataset["test_label"],
    )
