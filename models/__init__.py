"""
Module for retrieving use cases
to test client model verification.
"""

# External dependencies
from keras.models import Sequential

# Internal dependencies
from .mnist import get_mnist_model

def get_model(use_case: str) -> Sequential:
    """
    """
    model = None

    if use_case == 'mnist':
        model = get_mnist_model()

    return model
