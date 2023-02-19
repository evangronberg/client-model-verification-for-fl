"""
"""

# Python-native dependencies
from typing import Tuple, List

# External dependencies
from flwr.client import NumPyClient

# Internal dependencies
from model import Model

class Client(NumPyClient):
    """
    """
    def __init__(self, training_set: Tuple) -> None:
        """
        Arguments:
            train_set: The training dataset that will be
                       used for this particular client.
        """
        self.model = Model()
        self.train_set = training_set
