"""
"""

# Python-native dependencies
from typing import Tuple, List

# External dependencies
from flwr.client import NumPyClient
from keras.models import Sequential

class Client(NumPyClient):
    """
    """
    def __init__(self, model: Sequential, 
                 training_set: Tuple) -> None:
        """
        Arguments:
            model:     The model the client should use.
            train_set: The training dataset that will be
                       used for this particular client.
        """
        self.model = model
        self.training_set = training_set
