import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# flash_timer
from . import model
from . import tools


class ModelSetWeak:
    """A collection of weak-scaling FLASH models
    """
    def __init__(self,
                 model_set,
                 omp_threads,
                 leaf_blocks_per_rank,
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
        leaf_blocks_per_rank : [int]
            list of leaf blocks per MPI rank
        mpi_ranks : [int]
            list of MPI ranks used
        log_basename : str
            Prefix used in logfile name, e.g. 'sod3d' for 'sod3d_16.log`
        max_cores : int
            maximum cores available on node
        """
        self.model_set = model_set

        self.omp_threads = omp_threads
        self.leaf_blocks_per_rank = leaf_blocks_per_rank
        self.mpi_ranks = mpi_ranks
        self.max_cores = max_cores
        self.expand_sequences()

        self.models = {}

        for lbpr in self.leaf_blocks_per_rank:
            self.models[lbpr] = {}

            for ranks in self.mpi_ranks:
                leaf_blocks = lbpr * ranks
                self.models[lbpr][ranks] = model.Model(model_set=model_set,
                                                       omp_threads=self.omp_threads,
                                                       leaf_blocks=leaf_blocks,
                                                       mpi_ranks=ranks,
                                                       log_basename=log_basename)

    # =======================================================
    #                      Load/analysis
    # =======================================================
    def expand_sequences(self):
        """Expand sequence attributes
        """
        self.leaf_blocks_per_rank = tools.ensure_sequence(self.leaf_blocks_per_rank)

        if self.mpi_ranks is None:
            max_ranks = int(self.max_cores / self.omp_threads)
            self.mpi_ranks = tools.expand_power_sequence(largest=max_ranks)

        elif isinstance(self.mpi_ranks, int):
            self.mpi_ranks = tools.expand_power_sequence(largest=self.mpi_ranks)

    def get_times(self, unit, lbpr):
        """Return array of times versus mpi_ranks
        """
        times = []
        models = self.models[lbpr]

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
    def plot_times(self, unit='evolution',
                   y_scale='linear', x_scale='log'):
        """Plot scaling
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for lbpr in self.leaf_blocks_per_rank:
            times = self.get_times(unit=unit, lbpr=lbpr)
            ax.plot(x, times, marker='o', label=lbpr)

        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Time (s)',
                     x_scale=x_scale, y_scale=y_scale)

        return fig

    def plot_efficiency(self, unit='evolution', x_scale='log'):
        """Plot scaling
        """
        fig, ax = plt.subplots()
        x = self.mpi_ranks

        for lbpr in self.leaf_blocks_per_rank:
            times = self.get_times(unit=unit, lbpr=lbpr)
            y = 100 * times[0] / times
            ax.plot(x, y, marker='o', label=lbpr)

        last_rank = x[-1]
        ax.plot([1, last_rank], [100, 100], ls='--', color='black')

        self._set_ax(ax=ax, x=x, x_label='MPI Ranks', y_label='Efficiency (%)',
                     x_scale=x_scale)

        return fig

    def _set_ax(self, ax, x,
                x_label, y_label,
                x_scale=None, y_scale=None):
        """Set axis properties
        """
        ax.legend(title='Leaf blocks / rank')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{self.model_set}, OMP_THREADS={self.omp_threads}')

        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(x)
        ax.tick_params(axis='x', which='minor', bottom=False)




