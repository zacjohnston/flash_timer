import numpy as np
import pandas as pd


def get_evolution(filepath):
    """Get evolution time from .log file

    Return: float

    parameters
    ----------
    filepath : str
        path to .log file
    """
    offset = 19  # line offset of evolution from line_0
    lines = read_log(filepath=filepath)
    line_0 = get_summary_line(filepath)

    evol = lines[line_0 + offset].split()

    return float(evol[3])


def get_summary_line(filepath):
    """Get line number of perf summary from .log file

    Return: int

    parameters
    ----------
    filepath : str
        path to .log file
    """
    lines = read_log(filepath=filepath)
    line_0 = None

    for i, line in enumerate(lines):
        if line == ' perf_summary: code performance summary statistics':
            line_0 = i
            break

    if line_0 is None:
        print(f'No summary found in log file: {filepath}')

    return line_0


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
