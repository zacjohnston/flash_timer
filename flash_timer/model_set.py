import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
                 mpi_ranks=None,
                 log_basename='sod3d',
                 max_cores=128):
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
        max_cores : int
            maximum cores available
        """
        self.model_set = model_set

        self.omp_threads = omp_threads
        self.leaf_blocks = leaf_blocks
        self.mpi_ranks = mpi_ranks
        self.max_cores = max_cores
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

    # =======================================================
    #                      Load/analysis
    # =======================================================
    def expand_sequences(self):
        """Expand sequence attributes
        """
        self.leaf_blocks = tools.ensure_sequence(self.leaf_blocks)

        if self.mpi_ranks is None:
            max_ranks = int(self.max_cores / self.omp_threads)
            max_leaf_blocks = max(self.leaf_blocks)

            largest_rank = min(max_ranks, max_leaf_blocks)
            self.mpi_ranks = tools.expand_power_sequence(largest=largest_rank)

        elif isinstance(self.mpi_ranks, int):
            self.mpi_ranks = tools.expand_power_sequence(largest=self.mpi_ranks)

    def get_times(self, unit, leaf_blocks):
        """Return array of times versus mpi_ranks
        """
        times = []
        models = self.models[leaf_blocks]

        for ranks, m in models.items():
            print(leaf_blocks, ranks)
            t = float(m.table.loc[unit, 'avg'])
            # t = float(m.table.loc[unit, 'avg'][1])
            # t = float(m.table.loc[unit, 'tot'])
            # t = float(m.table.loc[unit, 'tot'][1])
            times += [t]

        return np.array(times)

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_strong_speedup(self, unit='evolution', log=False):
        """Plot scaling
        """
        fig, ax = plt.subplots()
        # x = self.mpi_ranks*self.omp_threads
        x = self.mpi_ranks

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            ax.plot(x, times[0]/times, marker='o', label=leafs)

        last_rank = x[-1]
        ax.plot([1, last_rank], [1, last_rank], ls='--', color='black')

        # self._set_ax(ax=ax, x_label='MPI ranks', y_label='Speedup')
        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Speedup',
                     log=log, yticks=True)

        return fig

    def plot_strong_time(self, unit='evolution'):
        """Plot scaling
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            ax.plot(x, times, marker='o', label=leafs)

        # self._set_ax(ax=ax, x_label='MPI ranks', y_label='Time (s)')
        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Time (s)')

        return fig

    def _set_ax(self, ax, x, x_label, y_label, yticks=False, log=False):
        """Set axis properties
        """
        ax.legend(title='Leaf blocks')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.model_set}, OMP_THREADS={self.omp_threads}')

        if log:
            ax.loglog()
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.minorticks_off()

        ax.set_xticks(x)

        if yticks:
            ax.set_yticks(x)



