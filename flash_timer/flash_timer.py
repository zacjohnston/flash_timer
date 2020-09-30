import numpy as np
import pandas as pd


def read_log(filepath):
    """Load .log file as list of lines

    Return: [str]

    parameters
    ----------
    filepath : str
        path to .log file
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    return lines
