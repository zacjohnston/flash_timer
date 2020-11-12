import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# flash_timer
from . import model
from . import tools


class ModelSet:
    """A collection of performace-scaling FLASH models

    Two modes:
        - ModelSet('strong', model_set, omp, leaf_blocks, mpi_ranks)
        - ModelSet('weak', model_set, omp, leaf_blocks_per_Rank, mpi_ranks)
    """
    def __init__(self,
                 scaling_type,
                 model_set,
                 omp=None,
                 leaf_blocks=None,
                 leaf_blocks_per_rank=None,
                 mpi_ranks=None,
                 config=None,
                 leaf_blocks_per_max_ranks=None,
                 log_basename='sod3d',
                 max_cores=128,
                 block_size=12,
                 n_timesteps=100,
                 unit='evolution',
                 ):
        """

        parameters
        ----------
        scaling_type : 'strong' or 'weak'
            type of scaling test
        model_set : str
            name of model set/collection
        omp : [int] or int
            number of OpenMP threads used
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
        config : str
            basename of config file, e.g. 'amd' for 'config/amd.ini'
            (defaults to 'default')
        """
        self.scaling_type = scaling_type
        self.model_set = model_set
        self.omp = omp
        self.mpi_ranks = mpi_ranks
        self.leaf_blocks_per_rank = leaf_blocks_per_rank
        self.leaf_blocks = leaf_blocks
        self.leaf_blocks_per_max_ranks = leaf_blocks_per_max_ranks
        self.max_cores = max_cores
        self.log_basename = log_basename
        self.block_size = block_size
        self.n_timesteps = n_timesteps
        self.unit = unit
        self.models = {}
        self.data = {}

        if self.scaling_type not in ['strong', 'weak']:
            raise ValueError(f"scaling_type='{scaling_type}', must be 'strong' or 'weak'")

        self.config = None
        self.load_config(config=config)

        self.expand_sequences()
        self.load_models()
        self.extract_data(unit=self.unit)

    # =======================================================
    #                      Init/Loading
    # =======================================================
    def load_config(self, config=None):
        """Load config parameters from file

        parameters
        ----------
        config : str
        """
        config_dict = tools.load_config(name=config)
        plot_config = tools.load_config(name='plotting')

        # override any options from plotting.ini
        plot_config['plot'].update(config_dict['plot'])
        config_dict.update(plot_config)

        self.config = config_dict

        if (self.scaling_type == 'weak') and (self.leaf_blocks_per_rank is None):
            self.leaf_blocks_per_rank = self.config['params']['leaf_blocks_per_rank']

        if (self.scaling_type == 'strong') and (self.leaf_blocks_per_max_ranks is None):
            self.leaf_blocks_per_max_ranks = self.config['params'][
                                                            'leaf_blocks_per_max_ranks']

    def expand_sequences(self):
        """Expand sequence attributes
        """
        self.expand_omp()
        self.expand_mpi_ranks()

        if self.scaling_type == 'weak':
            self.leaf_blocks_per_rank = tools.ensure_sequence(self.leaf_blocks_per_rank)

        elif self.scaling_type == 'strong':
            self.expand_leaf_blocks()
            self.leaf_blocks_per_max_ranks = tools.ensure_sequence(
                                                        self.leaf_blocks_per_max_ranks)

    def expand_omp(self):
        """Expand omp sequence
        """
        if self.omp is None:
            max_threads = int(self.max_cores / 2)
            self.omp = tools.expand_power_sequence(largest=max_threads)
        elif isinstance(self.omp, int):
            self.omp = tools.expand_power_sequence(largest=self.omp)

    def expand_mpi_ranks(self):
        """Expand mpi_ranks sequences
        """
        mpi_ranks = {}
        for omp in self.omp:
            max_ranks = int(self.max_cores / omp)

            if self.mpi_ranks is None:
                mpi_ranks[omp] = tools.expand_power_sequence(largest=max_ranks)
            elif isinstance(self.mpi_ranks, int):
                mpi_ranks[omp] = tools.expand_power_sequence(largest=self.mpi_ranks)
            else:
                mpi_ranks[omp] = self.mpi_ranks

        self.mpi_ranks = mpi_ranks

    def expand_leaf_blocks(self):
        """Expand leaf_blocks sequences
        """
        leaf_blocks = {}
        for omp in self.omp:
            max_ranks = int(self.max_cores / omp)

            if self.leaf_blocks is None:
                leaf_blocks[omp] = max_ranks * np.array(self.leaf_blocks_per_max_ranks)
            else:
                leaf_blocks[omp] = tools.ensure_sequence(self.leaf_blocks)

        self.leaf_blocks = leaf_blocks

    def load_models(self):
        """Load all model timing data
        """
        for omp in self.omp:
            self.models[omp] = {}
            leaf_sequence = self.get_leaf_sequence(omp=omp)

            for leaf in leaf_sequence:
                self.models[omp][leaf] = {}
                leaf_blocks = self.get_leaf_blocks(leaf=leaf, omp=omp)

                for i, ranks in enumerate(self.mpi_ranks[omp]):
                    print(f'\rLoading {omp}_{leaf_blocks[i]}_{ranks}', end=10*' ')

                    self.models[omp][leaf][ranks] = model.Model(
                                                        model_set=self.model_set,
                                                        omp=omp,
                                                        leaf_blocks=leaf_blocks[i],
                                                        mpi_ranks=ranks,
                                                        log_basename=self.log_basename)
        print()

    def extract_data(self, unit):
        """Extract performace quantities from model tables
        """
        funcs = {'times': self.get_times,
                 'speedup': self.get_speedup,
                 'efficiency': self.get_efficiency,
                 'zupcs': self.get_zupcs,
                 }

        print('Extracting performance data')
        for key, func in funcs.items():
            self.data[key] = {}

            for omp in self.omp:
                self.data[key][omp] = {}
                leaf_sequence = self.get_leaf_sequence(omp)

                for leaf in leaf_sequence:
                    self.data[key][omp][leaf] = func(omp=omp, leaf=leaf, unit=unit)

    def extract_xarray(self):
        """Extract multi-dimensional table of model timing data
        """
        print('Extracting performance data')
        omp_dict = {}

        for omp, omp_set in self.models.items():
            leaf_dict = {}

            omp_set = self.models[1]
            for leaf, leaf_set in omp_set.items():
                mpi_dict = {}

                for mpi_ranks, m in leaf_set.items():
                    mpi_dict[mpi_ranks] = m.table.to_xarray()

                leaf_xr = xr.concat(mpi_dict.values(), dim='mpi_ranks')
                leaf_xr.coords['mpi_ranks'] = list(mpi_dict.keys())
                leaf_dict[leaf] = leaf_xr

            omp_xr = xr.concat(leaf_dict.values(), dim='leaf')
            omp_xr.coords['leaf'] = list(leaf_dict.keys())
            omp_dict[omp] = omp_xr

        full_xr = xr.concat(omp_dict.values(), dim='omp')
        full_xr.coords['omp'] = list(omp_dict.keys())

        return full_xr

    # =======================================================
    #                      Analysis
    # =======================================================
    def get_times(self, omp, leaf, unit=None):
        """Return array of runtimes versus MPI ranks
        """
        times = []
        models = self.models[omp][leaf]

        if unit is None:
            unit = self.config['params']['unit']

        for ranks, m in models.items():
            t = float(m.table.loc[unit, 'avg'])
            # t = float(m.table.loc[unit, 'avg'][1])
            # t = float(m.table.loc[unit, 'tot'])
            # t = float(m.table.loc[unit, 'tot'][1])
            times += [t]

        return np.array(times)

    def get_zupcs(self, omp, leaf, unit=None):
        """Return array of Zone Updates Per Core Second, versus MPI ranks
        """
        times = self.get_times(omp=omp, unit=unit, leaf=leaf)
        leaf_blocks = self.get_leaf_blocks(leaf=leaf, omp=omp)

        zone_updates = self.n_timesteps * leaf_blocks * self.block_size**3
        core_seconds = omp * self.mpi_ranks[omp] * times
        zupcs = zone_updates / core_seconds

        return zupcs

    def get_efficiency(self, omp, leaf, unit=None):
        """Return array of scaling efficiency versus MPI ranks
        """
        times = self.get_times(omp=omp, unit=unit, leaf=leaf)

        eff_factor = {'strong': self.mpi_ranks[omp],
                      'weak': 1.0}.get(self.scaling_type)

        efficiency = 100 * times[0] / (eff_factor * times)

        return efficiency

    def get_speedup(self, omp, leaf, unit):
        """Return array of speedup versus MPI ranks
        """
        times = self.get_times(omp=omp, unit=unit, leaf=leaf)
        speedup = times[0] / times

        return speedup

    def get_model_table(self, omp, leaf, mpi_ranks):
        """Return timing table of specific model
        """
        m = self.models[omp][leaf][mpi_ranks]
        return m.table

    def get_leaf_sequence(self, omp):
        """Return sequence of leaf variable according to scaling_type
        """
        if self.scaling_type == 'strong':
            return self.leaf_blocks[omp]
        else:
            return self.leaf_blocks_per_rank

    def get_leaf_blocks(self, leaf, omp):
        """Return array of leaf_blocks versus MPI ranks
        """
        if self.scaling_type == 'strong':
            n_runs = len(self.mpi_ranks[omp])
            return np.full(n_runs, leaf)
        else:
            return leaf * self.mpi_ranks[omp]

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_multiple(self, omp=None, plots=None,
                      unit=None, x_scale=None, y_scale=None,
                      sub_figsize=(5, 3)):
        """Plot multiple sets of models

        parameters
        ----------
        omp : [int]
        plots : [str]
            types of plots
        unit : str
        x_scale : str
        y_scale : str
        sub_figsize : [width, height]
        """
        if omp is None:
            omp = self.omp
        else:
            omp = tools.ensure_sequence(omp)
        if plots is None:
            plots = self.config['plot']['multiplot'][self.scaling_type]

        nrows = len(omp)
        ncols = len(plots)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(sub_figsize[0]*ncols, sub_figsize[1]*nrows))

        for i, threads in enumerate(omp):
            for j, plot in enumerate(plots):
                ax = axes[i, j]
                self.plot(omp=threads, y_var=plot, ax=ax,
                          unit=unit, data_only=True)

                self._set_ax_subplot(axes=axes, row=i, col=j, omp=threads,
                                     x_var='mpi_ranks', y_var=plot,
                                     x_scale=x_scale, y_scale=y_scale)
        plt.tight_layout()
        return fig

    def plot(self, omp, y_var, unit=None, x_scale=None,
             ax=None, data_only=False):
        """Plot scaling
        """
        fig, ax = self._setup_fig_ax(ax=ax)
        x = self.mpi_ranks[omp]
        last_rank = x[-1]

        leaf_sequence = self.get_leaf_sequence(omp=omp)

        for leaf in leaf_sequence:
            y = self.data[y_var][omp][leaf]
            ax.plot(x, y, marker='o', label=leaf)

        if y_var == 'efficiency':
            ax.plot([1, last_rank], [100, 100], ls='--', color='black')
        elif y_var == 'speedup':
            ax.plot([1, last_rank], [1, last_rank], ls='--', color='black')

        self._set_ax(ax=ax, x_var='mpi_ranks', y_var=y_var, x=x, omp=omp,
                     x_scale=x_scale, data_only=data_only)

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

    def _set_ax(self, ax, x, x_var, y_var, omp,
                x_scale=None, y_scale=None,
                data_only=False):
        """Set axis properties
        """
        if not data_only:
            self._set_ax_legend(ax=ax)
            self._set_ax_title(ax=ax)
            self._set_ax_labels(ax=ax, x_var=x_var, y_var=y_var)
            self._set_ax_scale(ax=ax, x_var=x_var, y_var=y_var,
                               x_scale=x_scale, y_scale=y_scale)
            self._set_ax_xticks(ax=ax, x=x)
            self._set_ax_text(ax=ax, omp=omp)

    def _set_ax_subplot(self, axes, x_var, y_var, row, col, omp,
                        x_scale, y_scale):
        """Set axis properties for subplot (see plot_multiple)
        """
        ax = axes[row, col]
        nrows = axes.shape[0]
        ncols = axes.shape[1]

        if col == 0:
            self._set_ax_text(ax=ax, omp=omp)
            if self.scaling_type == 'strong':
                self._set_ax_legend(ax=ax)

            if row == 0:
                ax.set_title(f'{self.model_set}')
                if self.scaling_type == 'weak':
                    self._set_ax_legend(ax=ax)

        if row == nrows - 1:
            ax.set_xlabel(self.config['plot']['labels'][x_var])

        ax.set_ylabel(self.config['plot']['labels'][y_var])

        self._set_ax_scale(ax=ax, x_var=x_var, y_var=y_var,
                           x_scale=x_scale, y_scale=y_scale)
        self._set_ax_xticks(ax=ax, x=self.mpi_ranks[omp])

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

    def _set_ax_text(self, ax, omp):
        """Set axis text
        """
        ax.text(0.95, 0.05, f'OMP threads = {omp}',
                verticalalignment='bottom', horizontalalignment='right',
                fontsize=12, transform=ax.transAxes)

    def _set_ax_labels(self, ax, x_var, y_var):
        """Set axis labels

        parameters
        ----------
        ax : Axes
        x_var : str
        y_var : str
        """
        def get_label(key):
            return self.config['plot']['labels'].get(key, key)

        ax.set_xlabel(get_label(x_var))
        ax.set_ylabel(get_label(y_var))

    def _set_ax_xticks(self, ax, x):
        """Set axis ticks
        """
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(x)
        ax.tick_params(axis='x', which='minor', bottom=False)

    def _set_ax_scale(self, ax, x_var, y_var, x_scale, y_scale):
        """Set axis scales
        """
        def get_scale(var):
            for scale in ['log', 'linear']:
                if var in self.config['plot']['ax_scales'][scale]:
                    return scale
            return 'linear'

        if y_var == 'speedup':
            x_scale = 'linear'

        if x_scale is None:
            x_scale = get_scale(x_var)
        if y_scale is None:
            y_scale = get_scale(y_var)

        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)

