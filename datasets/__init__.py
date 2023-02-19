"""
"""

from .mnist import MNIST
from .cifar10 import CIFAR10

def get_dataset(dataset_name: str, n_clients: int,
                n_bad_clients: int):
    """
    """
    if dataset_name == 'mnist':
        dataset = MNIST(n_clients, n_bad_clients)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(n_clients, n_bad_clients)

    return dataset
