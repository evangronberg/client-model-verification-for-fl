"""
"""

# External dependencies
from keras.optimizers import Adam
from keras.metrics import accuracy
from keras.activations import relu
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.losses import categorical_crossentropy

def get_mnist_nn() -> None:
    """
    Arguments:
        None
    """
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=relu),
        Dense(10)
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss=categorical_crossentropy,
        metrics=[accuracy]
    )
    return model
