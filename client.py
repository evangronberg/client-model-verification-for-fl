"""
Module for defining a federated learning client.
"""

# Python-native dependencies
from typing import Tuple, Dict

# External dependencies
from keras.models import Sequential
from keras.callbacks import History
from flwr.client import NumPyClient
from flwr.common import Scalar, NDArrays

class Client(NumPyClient):
    """
    A client in a federated learning group.
    """
    def __init__(self, model: Sequential, 
                 training_set: Tuple, n_epochs: int,
                 batch_size: int) -> None:
        """
        Arguments:
            model:     The model the client should use.
            train_set: The training dataset that will be
                       used for this particular client.
        """
        self.model = model
        self.x_train = training_set[0]
        self.y_train = training_set[1]
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        """
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) \
            -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        """
        # Initialize the client model with the weights received from the server
        self.model.set_weights(parameters)

        # Update the model by training the server model on the client's data
        fit_history : History = self.model.fit(
            self.x_train, self.y_train,
            epochs=self.n_epochs,
            batch_size=self.batch_size
        )
        # Collect and send to the server the client results
        updated_parameters = self.model.get_weights()
        train_set_size = len(self.x_train)
        metrics = {
            'loss': fit_history.history['loss'][-1],
            'accuracy': fit_history.history['accuracy'][-1]
        }
        return updated_parameters, train_set_size, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) \
                 -> Tuple[float, int, Dict[str, Scalar]]:
        """
        TBA
        """
