import numpy as np
import pandas as pd


def extract_table(filepath):
    """Get perf timing table from .log file

    Return: pd.DataFrame

    parameters
    ----------
    filepath : str
        path to .log file
    """
    offset = 19  # line offset of evolution from line_0
    line_0 = get_summary_line(filepath)

    table = pd.read_csv(filepath, skiprows=line_0+offset, skipfooter=2,
                        header=None, sep=r"[ ]{2,}", engine='python')

    table.set_index(0, inplace=True)
    # table = table.transpose()
    table.columns = ['max', 'min', 'avg', 'calls']
    table.index.rename('unit', inplace=True)

    return table


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
    line_0 = []

    for i, line in enumerate(lines):
        if line == ' perf_summary: code performance summary statistics':
            line_0 += [i]

    if len(line_0) is 0:
        print(f'No performance summary found in log file: {filepath}')
        return None
    elif len(line_0) > 1:
        print(f'WARNING: multiple runs found in log file. Returning most recent only')
        return line_0[-1]
    else:
        return line_0[0]


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
