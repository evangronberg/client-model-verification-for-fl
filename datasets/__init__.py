"""
Module for retrieving use cases
to test client model verification.
"""

# Internal dependencies
from .mnist import MNIST
from .base import CMVDataset

def get_dataset(dataset_name: str, n_clients: int,
                n_bad_clients: int = None,
                n_scrambled_labels: int = None) -> CMVDataset:
    """
    """
    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST(
            n_clients, n_bad_clients,
            n_scrambled_labels
        )

    return dataset
