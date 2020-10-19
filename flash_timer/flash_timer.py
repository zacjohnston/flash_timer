import numpy as np
import pandas as pd


def get_summary_line(filepath):
    """Extract line number of perf summary from .log file
    Return: pd.DataFrame

    parameters
    ----------
    filepath : str
        path to .log file
    """
    lines = read_log(filepath=filepath)
    n = None

    for i, line in enumerate(lines):
        if line == ' perf_summary: code performance summary statistics':
            n = i
            break

    if n is None:
        print(f'No summary found in log file: {filepath}')
        
    return n


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
