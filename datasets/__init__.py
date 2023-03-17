"""
Module for getting datasets to test client model
verification for federated learning.
"""

from .mnist import MNIST
from .cifar10 import CIFAR10

def get_dataset(dataset_name: str, n_clients: int,
                n_bad_clients: int):
    """
    Gets the specified dataset, dividing it into n_clients partitions,
    n_bad_clients of which will be scrambled to simulate malicious data.

    Arguments:
        TBA
    Return Values:
        dataset: The requested dataset divided
                 into its requested partitions.
    """
    if dataset_name == 'mnist':
        dataset = MNIST(n_clients, n_bad_clients)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(n_clients, n_bad_clients)

    return dataset
