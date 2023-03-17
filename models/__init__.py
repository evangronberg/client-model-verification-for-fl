"""
Module for getting neural networks to test
client modek verification for federated learning.
"""

# External dependencies
from keras.models import Sequential

# Internal dependencies
from .mnist_nn import get_mnist_nn
from .cifar10_cnn import get_cifar10_cnn

def get_model(model_name: str) -> Sequential:
    """
    Gets the requested model.

    Arguments:
        model_name: The name of the model to get.
    Return Values:
        model:      The model.
    """
    if model_name == 'mnist':
        model = get_mnist_nn()
    if model_name == 'cifar10':
        model = get_cifar10_cnn()

    return model
