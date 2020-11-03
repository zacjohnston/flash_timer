import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# flash_timer
from . import model
from . import tools


class ModelSet:
    """A collection of performace-scaling FLASH models

    Two modes:
        - ModelSet('strong', model_set, omp_threads, leaf_blocks, mpi_ranks)
        - ModelSet('weak', model_set, omp_threads, leaf_blocks_per_Rank, mpi_ranks)
    """
    def __init__(self,
                 scaling_type,
                 model_set,
                 omp_threads=None,
                 leaf_blocks=None,
                 leaf_blocks_per_rank=(2, 4, 8),
                 mpi_ranks=None,
                 leaf_blocks_per_max_ranks=(2, 4, 8),
                 log_basename='sod3d',
                 max_cores=128,
                 block_size=12,
                 n_timesteps=100,
                 ):
        """

        parameters
        ----------
        scaling_type : 'strong' or 'weak'
            type of scaling test
        model_set : str
            name of model set/collection
        omp_threads : [int] or int
            OpenMP threads used
        leaf_blocks : [int] or int
            (strong only) list of total leaf blocks
        leaf_blocks_per_rank : [int] or int
            (weak only) list of leaf blocks per MPI rank
        mpi_ranks : [int] or int
            list of MPI ranks used
        leaf_blocks_per_max_ranks : [int]
            (strong only) number of leaf blocks per maximum rank number
        log_basename : str
            Prefix used in logfile name, e.g. 'sod3d' for 'sod3d_16.log`
        max_cores : int
            maximum cores available on node
        block_size : int
            block side-length in zones (assumes symmetric in x,y,z)
        """
        self.scaling_type = scaling_type
        self.model_set = model_set
        self.omp_threads = omp_threads
        self.mpi_ranks = mpi_ranks
        self.max_cores = max_cores
        self.log_basename = log_basename
        self.block_size = block_size
        self.n_timesteps = n_timesteps

        if self.scaling_type == 'weak':
            self.leaf_blocks_per_rank = leaf_blocks_per_rank
            self.leaf_blocks = None
            self.leaf_blocks_per_max_ranks = None

        elif self.scaling_type == 'strong':
            self.leaf_blocks_per_rank = None
            self.leaf_blocks = leaf_blocks
            self.leaf_blocks_per_max_ranks = leaf_blocks_per_max_ranks

        else:
            raise ValueError("scaling_type must be 'strong' or 'weak'")

        self.expand_sequences()

        self.models = {}
        self.load_models()

    # =======================================================
    #                      Init/Loading
    # =======================================================
    def expand_sequences(self):
        """Expand sequence attributes
        """
        self.expand_omp_threads()
        self.expand_mpi_ranks()

        if self.scaling_type == 'weak':
            self.leaf_blocks_per_rank = tools.ensure_sequence(self.leaf_blocks_per_rank)

        elif self.scaling_type == 'strong':
            self.expand_leaf_blocks()
            self.leaf_blocks_per_max_ranks = tools.ensure_sequence(
                                                        self.leaf_blocks_per_max_ranks)

    def expand_omp_threads(self):
        """Expand omp_threads sequence
        """
        if self.omp_threads is None:
            max_threads = int(self.max_cores / 2)
            self.omp_threads = tools.expand_power_sequence(largest=max_threads)
        elif isinstance(self.omp_threads, int):
            self.omp_threads = tools.expand_power_sequence(largest=self.omp_threads)

    def expand_mpi_ranks(self):
        """Expand mpi_ranks sequences
        """
        mpi_ranks = {}
        for threads in self.omp_threads:
            max_ranks = int(self.max_cores / threads)

            if self.mpi_ranks is None:
                mpi_ranks[threads] = tools.expand_power_sequence(largest=max_ranks)
            elif isinstance(self.mpi_ranks, int):
                mpi_ranks[threads] = tools.expand_power_sequence(largest=self.mpi_ranks)
            else:
                mpi_ranks[threads] = self.mpi_ranks

        self.mpi_ranks = mpi_ranks

    def expand_leaf_blocks(self):
        """Expand leaf_blocks sequences
        """
        leaf_blocks = {}
        for threads in self.omp_threads:
            max_ranks = int(self.max_cores / threads)

            if self.leaf_blocks is None:
                leaf_blocks[threads] = max_ranks * np.array(self.leaf_blocks_per_max_ranks)
            else:
                leaf_blocks[threads] = tools.ensure_sequence(self.leaf_blocks)

        self.leaf_blocks = leaf_blocks

    def load_models(self):
        """Load all model timing data
        """
        for threads in self.omp_threads:
            self.models[threads] = {}

            if self.scaling_type == 'strong':
                leaf_sequence = self.leaf_blocks[threads]
            else:
                leaf_sequence = self.leaf_blocks_per_rank

            for leaf in leaf_sequence:
                self.models[threads][leaf] = {}

                for ranks in self.mpi_ranks[threads]:
                    leaf_blocks = {'strong': leaf,
                                   'weak': leaf * ranks,
                                   }.get(self.scaling_type)
                    print(f'\rLoading {threads}_{leaf_blocks}_{ranks}', end=10*' ')

                    self.models[threads][leaf][ranks] = model.Model(
                                                        model_set=self.model_set,
                                                        omp_threads=threads,
                                                        leaf_blocks=leaf_blocks,
                                                        mpi_ranks=ranks,
                                                        log_basename=self.log_basename
                                                        )

    # =======================================================
    #                      Analysis
    # =======================================================
    def get_times(self, omp_threads, leaf, unit=None):
        """Return array of runtimes versus MPI ranks
        """
        times = []
        models = self.models[omp_threads][leaf]

        if unit is None:
            unit = 'evolution'

        for ranks, m in models.items():
            t = float(m.table.loc[unit, 'avg'])
            # t = float(m.table.loc[unit, 'avg'][1])
            # t = float(m.table.loc[unit, 'tot'])
            # t = float(m.table.loc[unit, 'tot'][1])
            times += [t]

        return np.array(times)

    def get_zupcs(self, omp_threads, leaf, unit=None):
        """Return array of Zone Updates Per Core Second, versus MPI ranks
        """
        times = self.get_times(omp_threads=omp_threads, unit=unit, leaf=leaf)

        if self.scaling_type == 'strong':
            leaf_blocks = leaf
        else:
            leaf_blocks = leaf * self.mpi_ranks[omp_threads]

        zone_updates = self.n_timesteps * leaf_blocks * self.block_size**3
        core_seconds = omp_threads * self.mpi_ranks[omp_threads] * times
        zupcs = zone_updates / core_seconds

        return zupcs

    def get_efficiency(self, omp_threads, leaf, unit=None):
        """Return array of scaling efficiency versus MPI ranks
        """
        times = self.get_times(omp_threads=omp_threads, unit=unit, leaf=leaf)

        if self.scaling_type == 'strong':
            eff_factor = self.mpi_ranks[omp_threads]
        else:
            eff_factor = 1.0

        efficiency = 100 * times[0] / (eff_factor * times)

        return efficiency

    def get_speedup(self, omp_threads, leaf, unit):
        """Return array of speedup versus MPI ranks
        """
        times = self.get_times(omp_threads=omp_threads, unit=unit, leaf=leaf)
        speedup = times[0] / times

        return speedup

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_multiple(self, omp_threads=None, plots=None,
                      unit=None, x_scale='log', y_scale='linear',
                      sub_figsize=(5, 3)):
        """Plot multiple sets of models

        parameters
        ----------
        omp_threads : [int]
        plots : [str]
            types of plots
        unit : str
        x_scale : str
        y_scale : str
        sub_figsize : [width, height]
        """
        plot_types = {'strong': ('speedup', 'efficiency'),
                      'weak': ('times', 'efficiency')}

        plot_funcs = {'times': self.plot_times,
                      'efficiency': self.plot_efficiency,
                      'speedup': self.plot_speedup}

        ylabels = {'times': 'Time (s)',
                   'efficiency': 'Efficiency (%)',
                   'speedup': 'Speedup'}

        if omp_threads is None:
            omp_threads = self.omp_threads
        if plots is None:
            plots = plot_types[self.scaling_type]

        nrows = len(omp_threads)
        ncols = len(plots)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(sub_figsize[0]*ncols, sub_figsize[1]*nrows))

        for i, threads in enumerate(omp_threads):
            for j, plot in enumerate(plots):
                ax = axes[i, j]
                plot_func = plot_funcs[plot]
                plot_func(omp_threads=threads, ax=ax, unit=unit, data_only=True)

                self._set_ax_subplot(axes=axes, row=i, col=j, omp_threads=threads,
                                     y_label=ylabels[plot],
                                     x_scale='linear' if plot == 'speedup' else x_scale,
                                     y_scale='linear' if plot == 'speedup' else y_scale)

        plt.tight_layout()
        return fig

    def plot_times(self, omp_threads, unit=None,
                   y_scale='linear', x_scale='log',
                   ax=None, data_only=False):
        """Plot scaling

        parameters
        ----------
        omp_threads : int
        unit : str
        y_scale : str
        x_scale : str
        ax : Axis
        data_only : bool
        """
        fig, ax = self._setup_fig_ax(ax=ax)
        x = self.mpi_ranks[omp_threads]

        if self.scaling_type == 'strong':
            leaf_sequence = self.leaf_blocks[omp_threads]
        else:
            leaf_sequence = self.leaf_blocks_per_rank

        for leaf in leaf_sequence:
            times = self.get_times(omp_threads=omp_threads, unit=unit, leaf=leaf)
            ax.plot(x, times, marker='o', label=leaf)

        self._set_ax(ax=ax, x=x, omp_threads=omp_threads,
                     x_label='MPI Ranks', y_label='Time (s)',
                     x_scale=x_scale, y_scale=y_scale,
                     data_only=data_only)

        return fig

    def plot_efficiency(self, omp_threads, unit=None, x_scale='log',
                        ax=None, data_only=False):
        """Plot scaling
        """
        fig, ax = self._setup_fig_ax(ax=ax)
        x = self.mpi_ranks[omp_threads]
        last_rank = x[-1]

        if self.scaling_type == 'strong':
            leaf_sequence = self.leaf_blocks[omp_threads]
        else:
            leaf_sequence = self.leaf_blocks_per_rank

        for leaf in leaf_sequence:
            y = self.get_efficiency(omp_threads=omp_threads, leaf=leaf, unit=unit)
            ax.plot(x, y, marker='o', label=leaf)

        ax.plot([1, last_rank], [100, 100], ls='--', color='black')

        self._set_ax(ax=ax, x=x, omp_threads=omp_threads,
                     x_label='MPI Ranks', y_label='Efficiency (%)',
                     x_scale=x_scale, data_only=data_only)

        return fig

    def plot_speedup(self, omp_threads, unit=None,
                     x_scale='linear', y_scale='linear',
                     ax=None, data_only=False):
        """Plot strong scaling speedup
        """
        if self.scaling_type == 'weak':
            raise ValueError("Can only plot speedup for strong scaling")

        fig, ax = self._setup_fig_ax(ax=ax)
        x = self.mpi_ranks[omp_threads]

        for leaf in self.leaf_blocks[omp_threads]:
            y = self.get_speedup(omp_threads=omp_threads, leaf=leaf, unit=unit)
            ax.plot(x, y, marker='o', label=leaf)

        last_rank = x[-1]
        ax.plot([1, last_rank], [1, last_rank], ls='--', color='black')

        self._set_ax(omp_threads=omp_threads, ax=ax, x=x,
                     x_label='MPI Ranks', y_label='Speedup',
                     x_scale=x_scale, y_scale=y_scale,
                     data_only=data_only)

        return fig

    # =======================================================
    #                      Plot tools
    # =======================================================
    def _setup_fig_ax(self, ax):
        """Setup fig, ax, checking if ax already provided

        parameters
        ----------
        ax : Axes
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        return fig, ax

    def _set_ax(self, ax, x, omp_threads,
                x_label, y_label, x_scale=None, y_scale=None,
                data_only=False):
        """Set axis properties
        """
        if not data_only:
            self._set_ax_legend(ax=ax)
            self._set_ax_title(ax=ax)
            self._set_ax_labels(ax=ax, x_label=x_label, y_label=y_label)
            self._set_ax_scale(ax=ax, x_scale=x_scale, y_scale=y_scale)
            self._set_ax_xticks(ax=ax, x=x)
            self._set_ax_text(ax=ax, omp_threads=omp_threads)

    def _set_ax_subplot(self, axes, row, col, omp_threads,
                        x_scale, y_scale, y_label):
        """Set axis properties for subplot (see plot_multiple)
        """
        ax = axes[row, col]
        nrows = axes.shape[0]
        ncols = axes.shape[1]

        if col == 0:
            self._set_ax_text(ax=ax, omp_threads=omp_threads)
            if self.scaling_type == 'strong':
                self._set_ax_legend(ax=ax)

            if row == 0:
                ax.set_title(f'{self.model_set}, OMP_THREADS={omp_threads}')
                if self.scaling_type == 'weak':
                    self._set_ax_legend(ax=ax)

        if row == nrows - 1:
            ax.set_xlabel('MPI ranks')

        ax.set_ylabel(y_label)
        self._set_ax_scale(ax=ax, x_scale=x_scale, y_scale=y_scale)
        self._set_ax_xticks(ax=ax, x=self.mpi_ranks[omp_threads])

    def _set_ax_legend(self, ax):
        """Set axis legend
        """
        titles = {'strong': 'Leaf blocks',
                  'weak': 'Leaf blocks / rank'}
        ax.legend(title=titles[self.scaling_type])

    def _set_ax_title(self, ax):
        """Set axis title
        """
        ax.set_title(f'{self.model_set}')

    def _set_ax_text(self, ax, omp_threads):
        """Set axis text
        """
        ax.text(0.95, 0.05, f'OMP threads = {omp_threads}',
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=12, transform=ax.transAxes)

    def _set_ax_labels(self, ax, x_label, y_label):
        """Set axis labels
        """
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def _set_ax_xticks(self, ax, x):
        """Set axis ticks
        """
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(x)
        ax.tick_params(axis='x', which='minor', bottom=False)

    def _set_ax_scale(self, ax, x_scale, y_scale):
        """Set axis scales
        """
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)



