import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# flash_timer
from . import model
from . import tools


class ModelSetStrong:
    """A collection of strong-scaling FLASH models
    """
    def __init__(self,
                 model_set,
                 omp_threads,
                 leaf_blocks=None,
                 mpi_ranks=None,
                 log_basename='sod3d',
                 max_cores=128,
                 leaf_blocks_per_max_ranks=(2, 4, 8)):
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
            maximum cores available on node
        leaf_blocks_per_max_ranks : [int]
            number of leaf blocks per maximum rank number
        """
        self.model_set = model_set
        self.omp_threads = omp_threads
        self.leaf_blocks = leaf_blocks
        self.mpi_ranks = mpi_ranks
        self.max_cores = max_cores
        self.leaf_blocks_per_max_ranks = leaf_blocks_per_max_ranks
        self.models = {}

        self.expand_sequences()

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
        max_ranks = int(self.max_cores / self.omp_threads)

        if self.mpi_ranks is None:
            self.mpi_ranks = tools.expand_power_sequence(largest=max_ranks)
        elif isinstance(self.mpi_ranks, int):
            self.mpi_ranks = tools.expand_power_sequence(largest=self.mpi_ranks)

        if self.leaf_blocks is None:
            self.leaf_blocks = max_ranks * np.array(self.leaf_blocks_per_max_ranks)
        else:
            self.leaf_blocks = tools.ensure_sequence(self.leaf_blocks)

    def get_times(self, unit, leaf_blocks):
        """Return array of times versus mpi_ranks
        """
        times = []
        models = self.models[leaf_blocks]

        for ranks, m in models.items():
            t = float(m.table.loc[unit, 'avg'])
            # t = float(m.table.loc[unit, 'avg'][1])
            # t = float(m.table.loc[unit, 'tot'])
            # t = float(m.table.loc[unit, 'tot'][1])
            times += [t]

        return np.array(times)

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_speedup(self, unit='evolution', x_scale='linear', y_scale='linear'):
        """Plot strong scaling speedup
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            y = times[0] / times
            ax.plot(x, y, marker='o', label=leafs)

        last_rank = x[-1]
        ax.plot([1, last_rank], [1, last_rank], ls='--', color='black')

        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Speedup',
                     x_scale=x_scale, y_scale=y_scale)

        return fig

    def plot_efficiency(self, unit='evolution', x_scale='log', y_scale='linear'):
        """Plot strong scaling speedup
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            y = 100 * times[0] / (self.mpi_ranks * times)
            ax.plot(x, y, marker='o', label=leafs)

        last_rank = x[-1]
        ax.plot([1, last_rank], [100, 100], ls='--', color='black')

        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Efficiency (%)',
                     x_scale=x_scale, y_scale=y_scale)

        return fig

    def plot_times(self, unit='evolution', x_scale='log', y_scale='linear'):
        """Plot strong-scaling runtimes
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for leafs in self.leaf_blocks:
            times = self.get_times(unit=unit, leaf_blocks=leafs)
            ax.plot(x, times, marker='o', label=leafs)

        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Time (s)',
                     x_scale=x_scale, y_scale=y_scale)

        return fig

    def _set_ax(self, ax, x, x_label, y_label,
                x_scale=None, y_scale=None):
        """Set axis properties
        """
        ax.legend(title='Leaf blocks')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.model_set}, OMP_THREADS={self.omp_threads}')

        if x_scale == 'log':
            ax.set_xscale(x_scale)
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if y_scale == 'log':
            ax.set_yscale(y_scale)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_yticks(x)
            ax.tick_params(axis='y', which='minor', left=False)

        ax.set_xticks(x)
        ax.tick_params(axis='x', which='minor', bottom=False)


