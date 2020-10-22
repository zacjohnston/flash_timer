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
"""

import os


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


def log_filepath(run, model, model_set):
    """Return filepath to FLASH .log file

    Parameters
    ----------
    run : str
    model : str
    model_set : str
    """
    path = model_path(model, model_set=model_set)
    filename = f'{run}.log'

    return os.path.join(path, filename)

