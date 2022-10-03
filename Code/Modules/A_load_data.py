import numpy as np
from tensorflow.keras.datasets import mnist

from Code.Utilities.pair_utils import make_pairs

def load_data():

    # Load MNIST dataset
    print("[INFO] loading MNIST dataset...")
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    # Scale images so that they are in [0,1]
    train_X = train_X / 255.0
    test_X  = test_X / 255.0

    # Add a channel dimension to the images
    train_X = np.expand_dims(train_X, axis=-1)
    test_X  = np.expand_dims(test_X, axis=-1)

    # Prepare the positive and negative pairs
    print("[INFO] preparing positive and negative pairs...")
    (pair_train, label_train) = make_pairs(train_X, train_Y)
    (pair_test, label_test)   = make_pairs(test_X, test_Y)

    return (pair_train, label_train), (pair_test, label_test)