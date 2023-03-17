"""
Module for loading the CIFAR10 database in a format
conducive to testing client model verification.
"""

# Python-native dependencies
from random import shuffle

# External dependencies
from keras.utils import to_categorical
from keras.datasets.cifar10 import load_data

class CIFAR10():
    """
    The CIFAR10 dataset formatted for
    testing client model verification.
    """
    def __init__(self, n_clients: int,
                 n_bad_clients: int) -> None:
        """
        Arguments:
            n_clients:     The total number of clients.
            n_bad_clients: The number of clients that should
                           have a tampered training set.
        """
        # Load the data
        (x_train, y_train), (x_test, y_test) = load_data()

        # Normalize the inputs
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # One-hot encode the training labels
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Partition the data into as many
        # training sets as there are clients
        train_set_inputs = []
        train_set_outputs = []
        client_dataset_size = len(x_train) // n_clients
        for i in range(n_clients):
            train_set_inputs.append(
                x_train[i*client_dataset_size:(i+1)*client_dataset_size])
            train_set_outputs.append(
                y_train[i*client_dataset_size:(i+1)*client_dataset_size])

        # For each bad client, scramble one of
        # the training partition's labels
        for i in range(n_bad_clients):
            shuffle(train_set_outputs[i])

        # Collect and store the training sets for the clients
        self.training_sets = []
        for i in range(n_clients):
            self.training_sets.append((
                train_set_inputs[i], train_set_outputs[i]))

        # Collect and store the test set for the server
        self.test_set = (x_test, y_test)
