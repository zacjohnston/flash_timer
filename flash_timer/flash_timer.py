import numpy as np
import pandas as pd


def extract_table(filepath, loglines=None,
                  table_offset=8):
    """Get perf timing table from .log file

    Return: pd.DataFrame

    parameters
    ----------
    filepath : str
        path to .log file
    loglines : [str]
    table_offset : int
        offset (lines) of performance table from summary line
    """
    summary_line = get_summary_line(filepath=filepath, loglines=loglines)

    table = pd.read_csv(filepath, skiprows=summary_line + table_offset,
                        skipfooter=2, header=None,
                        sep=r"[ ]{2,}", engine='python')

    table.set_index(0, inplace=True)
    # table = table.transpose()
    table.columns = ['max', 'min', 'avg', 'calls']
    table.index.rename('unit', inplace=True)

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
    line_0 = []

    for i, line in enumerate(loglines):
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
