"""
Module for defining the structure of the convolutional 
neural network model used by the server and all clients.

Model taken from this article:
https://pythonistaplanet.com/cifar-10-image-classification-using-keras/
"""

# External dependencies
from keras import Sequential
from keras.optimizers import SGD
from keras.metrics import accuracy
from keras.constraints import maxnorm
from keras.activations import relu, softmax
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

class Model():
    """
    """
    def __init__(self) -> None:
        """
        Arguments:
            None
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), input_shape=(32, 32, 3),
                   activation=relu, padding='same'),
            Dropout(0.2),
            Conv2D(32, (3, 3), activation=relu, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation=relu, padding='same'),
            Dropout(0.2),
            Conv2D(64, (3, 3), activation=relu, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation=relu, padding='same'),
            Dropout(0.2),
            Conv2D(128, (3, 3), activation=relu, padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.2),
            Dense(1024, activation=relu, kernel_constraint=maxnorm(3)),
            Dropout(0.2),
            Dense(512, activation=relu, kernel_constraint=maxnorm(3)),
            Dropout(0.2),
            Dense(10, activation=softmax)
        ])
        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=SGD(),
            metrics=[accuracy]
        )
