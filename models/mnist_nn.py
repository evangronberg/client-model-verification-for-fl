"""
Module for defining a basic neural
network for use with the MNIST dataset.
"""

# External dependencies
from keras.optimizers import Adam
from keras.metrics import accuracy
from keras.activations import relu
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import categorical_crossentropy

def get_mnist_nn() -> Sequential:
    """
    Gets an basic neural network compatible with MNIST.

    Arguments:
        None
    Return Values:
        model: An MNIST-compatible neural network.
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=relu),
        Dense(10)
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss=categorical_crossentropy,
        metrics=[accuracy]
    )
    return model
