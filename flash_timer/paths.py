"""Standardised strings for paths and filenames for FLASH models

You must set the bash environment variable:
    - FLASH_MODELS: path to top directory containing FLASH models

Expected model directory structure
----------------------------------
$FLASH_MODELS
│
└───<model_set>
|   |
|   └───<model>
|   │   │   <run>.dat
|   │   │   <run>.log
|   │   │   ...
|   │   │
|   │   └───output
|   │       │   <run>_hdf5_chk_0000
|   │       │   <run>_hdf5_chk_0001
|   │       │   ...

where: model="{model_set}_{omp_threads}_{leaf_blocks}_{mpi_ranks}"
       run="{model_set}_{leaf_blocks}"
"""

import os


# ===============================================================
#                      FlashTimer
# ===============================================================
def flash_timer_path():
    """Return path to flash_timer repo
    """
    try:
        path = os.environ['FLASH_TIMER']
    except KeyError:
        raise EnvironmentError('Environment variable FLASH_TIMER not set. '
                               'Set path to flash_timer directory, e.g., '
                               "'export FLASH_TIMER=${HOME}/codes/flash_timer'")
    return path


def config_filepath(name):
    """Return path to config file

    parameters
    ----------
    name : str
        basename of config file
    """
    path = flash_timer_path()
    return os.path.join(path, 'flash_timer', 'config', f'{name}.ini')


# ===============================================================
#                      FLASH files
# ===============================================================
def model_path(model, model_set):
    """Return path to model directory

    parameters
    ----------
    model : str
    model_set : str
    """
    try:
        flash_models_path = os.environ['FLASH_MODELS']
    except KeyError:
        raise EnvironmentError('Environment variable FLASH_MODELS not set. '
                               'Set path to directory containing flash models, e.g., '
                               "'export FLASH_MODELS=${HOME}/BANG/runs'")

    return os.path.join(flash_models_path, model_set, model)


def log_filepath(model_set, leaf_blocks, mpi, omp,
                 basename='sod3d'):
    """Return filepath to FLASH .log file

    Parameters
    ----------
    model_set : str
    leaf_blocks : int
    mpi : int
    omp : int
    basename : str
    """
    model = f'{model_set}_{omp}_{leaf_blocks}_{mpi}'
    path = model_path(model=model, model_set=model_set)
    filename = f'{basename}_{leaf_blocks}.log'

    return os.path.join(path, filename)

