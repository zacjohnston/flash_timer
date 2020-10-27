import numpy as np
import matplotlib.pyplot as plt

# flash_timer
from . import model
from . import tools


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
        omp_threads : int
            OpenMP threads used
        leaf_blocks : [int]
            list of total leaf blocks
        mpi_ranks : [int]
            list of MPI ranks used
        log_basename : str
            Prefix used in logfile name, e.g. 'sod3d' for 'sod3d_16.log`
        """
        self.model_set = model_set

        self.omp_threads = omp_threads
        self.leaf_blocks = leaf_blocks
        self.mpi_ranks = mpi_ranks
        self.expand_sequences()

        self.models = {}

        for leafs in self.leaf_blocks:
            self.models[leafs] = {}

            for ranks in self.mpi_ranks:
                self.models[leafs][ranks] = model.Model(model_set=model_set,
                                                        omp_threads=self.omp_threads,
                                                        leaf_blocks=leafs,
                                                        mpi_ranks=ranks,
                                                        log_basename=log_basename)

    def expand_sequences(self):
        """Expand sequence attributes
        """
        self.leaf_blocks = tools.ensure_sequence(self.leaf_blocks)

        if not isinstance(self.mpi_ranks, (list, tuple, np.ndarray)):
            self.mpi_ranks = tools.expand_power_sequence(largest=self.mpi_ranks)

    def get_times(self, unit, leaf_blocks):
        """Return array of times versus mpi_ranks
        """
        times = []
        models = self.models[leaf_blocks]

        for i, m in models.items():
            t = m.table.loc[unit, 'avg']
            # t = m.table.loc[unit, 'avg'][1]
            times += [t]

        return np.array(times)

    def plot_strong_speedup(self, unit='evolution'):
        """Plot scaling
        """
        fig, ax = plt.subplots()

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            ax.plot(self.mpi_ranks, times[0]/times, marker='o', label=leafs)

        last_rank = self.mpi_ranks[-1]
        ax.plot([0, last_rank], [0, last_rank], ls='--', color='black')

        self._set_ax(ax=ax, x_label='MPI ranks', y_label='Speedup')

    def plot_strong_time(self, unit='evolution'):
        """Plot scaling
        """
        fig, ax = plt.subplots()

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            ax.plot(self.mpi_ranks, times, marker='o', label=leafs)

        self._set_ax(ax=ax, x_label='MPI ranks', y_label='Time (s)')

    def _set_ax(self, ax, x_label, y_label):
        """Set axis properties
        """
        ax.legend(title='Leaf blocks')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.model_set}, OMP_THREADS={self.omp_threads}')


