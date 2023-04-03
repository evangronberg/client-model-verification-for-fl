"""
Module for defining both the MNIST
dataset and a compatiable neural network.

Drawn from the official Keras documentation:
https://keras.io/examples/vision/mnist_convnet/
"""

# Python-native dependencies
from random import shuffle

# External dependencies
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
from numpy import ndarray, expand_dims, argmax, array

# Internal dependencies
from .base import CMVDataset

class MNIST(CMVDataset):
    """
    The MNIST dataset formatted for
    testing client model verification.
    """ 
    def __init__(self, n_clients: int,
                 n_bad_clients: int = None) -> None:
        """
        Arguments:
            n_clients:     The total number of clients.
            n_bad_clients: The number of clients that should
                           have a tampered training set.
        """
        super().__init__()

        # Load the data
        dataset = load_data()
        x_train : ndarray = dataset[0][0]
        y_train : ndarray = dataset[0][1]
        x_test : ndarray = dataset[1][0]
        y_test : ndarray = dataset[1][1]

        # Normalize all input to the [0, 1] range
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # Shape each individual image to (28, 28, 1)
        x_train = expand_dims(x_train, -1)
        x_test = expand_dims(x_test, -1)

        # One-hot encode the outputs
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

        self.test_set = (x_test, y_test)

        # This is for when the server retrieves the test set
        if n_bad_clients is None:
            self.training_sets = []
            return

        train_set_inputs = []
        train_set_outputs = []
        client_dataset_size = len(x_train) // n_clients
        for i in range(n_clients):
            train_set_inputs.append(
                x_train[i*client_dataset_size:(i+1)*client_dataset_size])
            train_set_outputs.append(
                y_train[i*client_dataset_size:(i+1)*client_dataset_size])

        scrambles = []
        for i in range(n_bad_clients):
            scramble = [x for x in range(10)]
            shuffle(scramble)
            scrambles.append(scramble)

        for i in range(n_bad_clients):
            scrambled_train_set_outputs = []
            for output in train_set_outputs[i]:
                scrambled_train_set_outputs.append(to_categorical(
                    scrambles[i][argmax(output)], num_classes=10))
            train_set_outputs[i] = array(scrambled_train_set_outputs)

        self.training_sets = []
        for i in range(n_clients):
            self.training_sets.append(
                (train_set_inputs[i], train_set_outputs[i]))

        
