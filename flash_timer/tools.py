"""
Misc. general-use functions
"""

import numpy as np


def expand_power_sequence(largest=None, length=None):
    """Return sequence of powers-of-two

    examples: largest=16 returns [1,2,4,8,16]
              length=4 returns [1,2,3,4]

    parameters
    ----------
    largest : int
        largest power in sequence
    length : int
        length of sequence
    """
    if length is None:
        if largest is None:
            raise ValueError('Must provide one of: end or length')

        length = int(np.log2(largest)) + 1

    return np.full(length, 2)**np.arange(length)


def ensure_sequence(x):
    """Ensure given object is in the form of a sequence.
    If object is scalar, return as length-1 list.

    parameters
    ----------
    x : array or scalar
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    else:
        return [x, ]
