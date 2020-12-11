from . import logfile
from . import paths


class Model:
    """A single FLASH model
    """
    def __init__(self,
                 model_set,
                 omp,
                 leaf_blocks,
                 mpi,
                 log_filepath=None,
                 log_basename='sod3d',
                 which_table='summary',
                 ):
        """
        parameters
        ----------
        model_set : str
            name of model set/collection
        leaf_blocks : int
            total no. leaf blocks
        omp : int
            no. OpenMP threads used
        mpi : int
            no. MPI ranks used
        log_filepath : str
            path to .log file (optional)
        which_table : 'summary' or 'main'
            timing table to read from logfile,
                'summary': stats from all processes (not available when leaf_blocks=mpi)
                'main': stats for main process only
        """
        if log_filepath is None:
            log_filepath = paths.log_filepath(model_set=model_set,
                                              leaf_blocks=leaf_blocks,
                                              omp=omp,
                                              mpi=mpi,
                                              basename=log_basename)

        self.log_filepath = log_filepath
        self.model_set = model_set
        self.omp = omp
        self.leaf_blocks = leaf_blocks
        self.mpi = mpi
        self.table = logfile.extract_table(filepath=self.log_filepath,
                                           which_table=which_table)
