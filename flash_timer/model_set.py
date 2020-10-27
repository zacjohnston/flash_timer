import numpy as np
import matplotlib.pyplot as plt

# flash_timer
from . import model
from .tools import ensure_sequence


class ModelSet:
    """A collection of FLASH models
    """
    def __init__(self,
                 model_set,
                 omp_threads,
                 leaf_blocks,
                 mpi_ranks,
                 log_basename='sod3d'):
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
        log_basename : str
            Prefix used in logfile name, e.g. 'sod3d' for 'sod3d_16.log`
        """
        self.model_set = model_set
        self.omp_threads = ensure_sequence(omp_threads)
        self.leaf_blocks = ensure_sequence(leaf_blocks)
        self.mpi_ranks = ensure_sequence(mpi_ranks)

        self.models = {}

        for threads in self.omp_threads:
            self.models[threads] = {}

            for leafs in self.leaf_blocks:
                self.models[threads][leafs] = {}

                for ranks in self.mpi_ranks:
                    self.models[threads][leafs][ranks] = model.Model(
                                                            model_set=model_set,
                                                            omp_threads=threads,
                                                            leaf_blocks=leafs,
                                                            mpi_ranks=ranks,
                                                            log_basename=log_basename)

    def get_times(self, unit, omp_threads, leaf_blocks):
        """Return array of times versus mpi_ranks
        """
        times = []
        models = self.models[omp_threads][leaf_blocks]

        for i, m in models.items():
            t = m.table.loc[unit, 'avg']
            # t = m.table.loc[unit, 'avg'][1]
            times += [t]

        return np.array(times)

    def plot_strong_speedup(self, omp_threads, unit='evolution'):
        """Plot scaling
        """
        fig, ax = plt.subplots()

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, omp_threads=omp_threads,
                                   leaf_blocks=leafs)
            ax.plot(self.mpi_ranks, times[0]/times, marker='o', label=leafs)

        last_rank = self.mpi_ranks[-1]
        ax.plot([0, last_rank], [0, last_rank], ls='--', color='black')

        self._set_ax(ax=ax, omp_threads=omp_threads, x_label='MPI ranks',
                     y_label='Speedup')

    def plot_strong_time(self, omp_threads, unit='evolution'):
        """Plot scaling
        """
        fig, ax = plt.subplots()

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, omp_threads=omp_threads,
                                   leaf_blocks=leafs)
            ax.plot(self.mpi_ranks, times, marker='o', label=leafs)

        self._set_ax(ax=ax, omp_threads=omp_threads, x_label='MPI ranks',
                     y_label='Time (s)')

    def _set_ax(self, ax, omp_threads, x_label, y_label):
        """Set axis properties
        """
        ax.legend(title='Leaf blocks')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.model_set}, OMP_THREADS={omp_threads}')


