from . import logfile
from . import paths


class Model:
    """A single FLASH model
    """
    def __init__(self,
                 model_set,
                 omp_threads,
                 leaf_blocks,
                 mpi_ranks,
                 log_filepath=None,
                 log_basename='sod3d'):
        """
        parameters
        ----------
        model_set : str
            name of model set/collection
        leaf_blocks : int
            total no. leaf blocks
        omp_threads : int
            no. OpenMP threads used
        mpi_ranks : int
            no. MPI ranks used
        log_filepath : str
            path to .log file (optional)
        """
        if log_filepath is None:
            log_filepath = paths.log_filepath(model_set=model_set,
                                              leaf_blocks=leaf_blocks,
                                              omp_threads=omp_threads,
                                              mpi_ranks=mpi_ranks,
                                              basename=log_basename)

        self.log_filepath = log_filepath
        self.model_set = model_set
        self.omp_threads = omp_threads
        self.leaf_blocks = leaf_blocks
        self.mpi_ranks = mpi_ranks
        self.table = logfile.extract_table(filepath=self.log_filepath)
