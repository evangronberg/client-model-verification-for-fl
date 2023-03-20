"""
Module for defining datasets formatted
to test client model verification.
"""

# Python-native dependencies
from typing import Tuple, List

# External dependencies
from numpy import ndarray

class CMVDataset():
    """
    A dataset formatted to test client model verification.
    """
    def __init__(self) -> None:
        """
        Arguments:
            None
        """
        self.training_sets : List[Tuple[ndarray]] = []
        self.test_set : Tuple[ndarray, ndarray] = None
