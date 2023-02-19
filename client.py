"""
"""

# External dependencies
from flwr.client import NumPyClient

# Internal dependencies
from model import Model

class Client(NumPyClient):
    """
    """
    def __init__(self, train_set) -> None:
        """
        Arguments:
            train_set: The training dataset that will be
                       used for this particular client.
        """
        self.model = Model()
        self.train_set = train_set
