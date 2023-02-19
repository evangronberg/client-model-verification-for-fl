"""
Module for defining the structure of the neural
network model used by the server and all clients.
"""

# External dependencies
from keras import Sequential
from keras.layers import Input, Dense
from keras.activations import relu

class Model():
    """
    """
    def __init__(self) -> None:
        """
        Arguments:
            None
        """
        self.model = Sequential([
            Input(shape=()),
            Dense(3, activation=relu)
        ])
