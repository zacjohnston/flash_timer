"""
Misc. general-use functions
"""

import numpy as np


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
