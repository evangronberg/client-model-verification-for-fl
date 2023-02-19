"""
Module for loading the CIFAR-10 datase in a format
conducive to testing client model verification.
"""

# Python-native dependencies
from random import shuffle

# External dependencies
from keras.datasets.cifar10 import load_data

class Dataset():
    """
    """
    def __init__(self, n_good_clients: int,
                 n_bad_clients: int) -> None:
        """
        Arguments:
            n_good_clients: The number of clients that should
                            have an untampered training set.
            n_bad_clients:  The number of clients that should
                            have a tampered training set.
        """
        # Load the data
        (x_train, y_train), (_, _) = load_data()

        # Partition the data into as many
        # training sets as there are clients
        n_clients = n_good_clients + n_bad_clients
        client_dataset_size = len(x_train) // n_clients
        train_set_inputs = []
        train_set_outputs = []
        for i in range(n_clients):
            train_set_inputs.append(
                x_train[i*client_dataset_size:(i+1)*client_dataset_size])
            train_set_outputs.append(
                y_train[i*client_dataset_size:(i+1)*client_dataset_size])

        # For each bad client, scramble one of
        # the training partition's labels
        for i in range(n_bad_clients):
            shuffle(train_set_outputs[i])

        # Collect and store the training sets
        self.training_sets = []
        for i in range(n_clients):
            self.training_sets.append((
                train_set_inputs[i], train_set_outputs[i]))

def get_training_set(index: int):
    """
    """
    # TODO: Need to figure out a way to make this dynamic within Flower.
    dataset = Dataset(9, 1)
    return dataset.training_sets[index]
