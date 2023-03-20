"""
Module for defining both the MNIST
dataset and a compatiable neural network.

Drawn from the official Keras documentation:
https://keras.io/examples/vision/mnist_convnet/
"""

# External dependencies
from keras import Input
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

def get_mnist_model() -> Sequential:
    """
    Creates an MNIST-compatible neural network.
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(10, activation='softmax'),
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
