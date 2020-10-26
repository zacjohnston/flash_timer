from . import model


class ModelSet:
    """A collection of FLASH models
    """
    def __init__(self,
                 model_set,
                 omp_threads,
                 leaf_blocks,
                 mpi_ranks,
                 log_filepath=None):
        """

        parameters
        ----------
        model_set : str
            name of model set/collection
        leaf_blocks : [int]
            list of total leaf blocks
        omp_threads : [int]
            list of OpenMP threads used
        mpi_ranks : [int]
            list of MPI ranks used
        log_filepath : str
            path to .log file (optional)
        """
        self.model_set = model_set
        self.omp_threads = omp_threads
        self.leaf_blocks = leaf_blocks
        self.mpi_ranks = mpi_ranks

        self.models = {}

        for threads in self.omp_threads:
            self.models[threads] = {}

            for leaf in self.leaf_blocks:
                self.models[threads][leaf] = {}

                for rank in self.mpi_ranks:
                    self.models[threads][leaf][rank] = model.Model(model_set=model_set,
                                                                   omp_threads=threads,
                                                                   leaf_blocks=leaf,
                                                                   mpi_ranks=rank)



