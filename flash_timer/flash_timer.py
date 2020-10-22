import numpy as np
import pandas as pd


def extract_table(filepath, loglines=None):
    """Get perf timing table from .log file

    Return: pd.DataFrame

    parameters
    ----------
    filepath : str
        path to .log file
    loglines : [str]
    """
    table_line = get_table_line(filepath=filepath, loglines=loglines)

    table = pd.read_csv(filepath, skiprows=table_line, skipfooter=2,
                        header=None, sep=r"[ ]{2,}", engine='python',
                        names=['unit', 'max', 'min', 'avg', 'calls'])

    table.set_index('unit', inplace=True)
    # table = table.transpose()

    return table


def get_evolution(filepath=None, loglines=None, offset=19):
    """Get evolution time (avg/proc) from .log file

    Return: float

    parameters
    ----------
    filepath : str
        path to .log file
    loglines : [str]
    offset : int
        offset (lines) of evolution row from summary line
    """
    loglines = check_loglines(filepath=filepath, loglines=loglines)
    summary_line = get_summary_line(loglines=loglines)
    evol = loglines[summary_line + offset].split()

    return float(evol[3])


def get_table_line(filepath=None, loglines=None):
    """Get line number of perf summary table from .log file
    Note: Finds first row containing 'initialization'

    Return: int

    parameters
    ----------
    filepath : str
        path to .log file
    loglines: [str]
    """
    loglines = check_loglines(filepath=filepath, loglines=loglines)
    summary_line = get_summary_line(loglines=loglines)

    table_offset = []

    for i, line in enumerate(loglines[summary_line:]):
        if 'initialization' in line:
            table_offset = i
            break

    return summary_line + table_offset


def get_summary_line(filepath=None, loglines=None):
    """Get line number of perf summary from .log file

    Return: int

    parameters
    ----------
    filepath : str
        path to .log file
    loglines: [str]
    """
    loglines = check_loglines(filepath=filepath, loglines=loglines)
    summary_line = []

    for i, line in enumerate(loglines):
        if line == ' perf_summary: code performance summary statistics':
            summary_line += [i]

    if len(summary_line) is 0:
        print(f'No performance summary found in log file: {filepath}')
        return None
    elif len(summary_line) > 1:
        print(f'WARNING: multiple runs found in log file. Returning most recent only')
        return summary_line[-1]
    else:
        return summary_line[0]


def check_loglines(filepath, loglines):
    """Check if loglines provided, otherwise load from file
    """
    if loglines is None:
        if filepath is None:
            raise ValueError('Must provide either filepath or loglines')
        loglines = load_loglines(filepath=filepath)

    return loglines


def load_loglines(filepath):
    """Load .log file as list of lines

    Return: [str]

    parameters
    ----------
    filepath : str
        path to .log file
    """
    with open(filepath, 'r') as f:
        loglines = f.read().splitlines()

    return loglines
