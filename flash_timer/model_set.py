import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# flash_timer
from . import model
from . import tools


class ModelSet:
    """A collection of performace-scaling FLASH models

    Two scaling modes:
        - ModelSet('strong', model_set, omp, mpi, leaf_per_max_rank)
        - ModelSet('weak',   model_set, omp, mpi, leaf_per_rank)
    """
    def __init__(self,
                 scaling_type,
                 model_set,
                 omp=None,
                 mpi=None,
                 leaf_per_rank=None,
                 leaf_per_max_rank=None,
                 leaf=None,
                 config=None,
                 log_basename='sod3d',
                 max_cores=128,
                 block_size=12,
                 n_timesteps=100,
                 unit='evolution',
                 time_column='avg',
                 which_table='summary',
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
        leaf_per_max_rank : [int] or int
            leaf blocks per max mpi rank (strong only)
        leaf_per_rank : [int] or int
             leaf blocks per mpi rank (weak only)
        leaf : [int] or int
            specify leaf blocks (if fixed across OMP threads)
            overrides leaf_per_rank and leaf_per_max_rank
        mpi : [int] or int
            number of MPI ranks used
        log_basename : str
            Prefix used in logfile name, e.g. 'sod3d' for 'sod3d_16.log`
        max_cores : int
            maximum cores available on node
        block_size : int
            block side-length in zones (assumes symmetric in x,y,z)
        config : str
            basename of config file, e.g. 'amd' for 'config/amd.ini'
            (defaults to 'default')
        unit : str
            which timing unit in .log table to read from
        time_column : str
            name of column to use from logfile performance table
        which_table : 'summary' or 'main'
            timing table to read from logfile,
                'summary': stats from all processes (not available when leaf_blocks=mpi)
                'main': stats for main process only
        """
        self.scaling_type = scaling_type
        self.model_set = model_set
        self.omp = None
        self.mpi = {}
        self.leaf = {}
        self.leaf_per_max_rank = leaf_per_max_rank
        self.leaf_per_rank = leaf_per_rank
        self.max_cores = max_cores
        self.log_basename = log_basename
        self.block_size = block_size
        self.n_timesteps = n_timesteps
        self.unit = unit
        self.models = {}
        self.time_column = time_column
        self.which_table = which_table
        self.data = None

        if self.scaling_type not in ['strong', 'weak']:
            raise ValueError(f"scaling_type='{scaling_type}', must be 'strong' or 'weak'")

        self.config = None
        self.load_config(config=config)

        self.expand_omp(omp=omp)
        self.expand_mpi(mpi=mpi)
        self.expand_leaf(leaf=leaf)

        self.load_models()
        self.extract_data()

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

        if self.scaling_type == 'strong':
            if self.leaf_per_max_rank is None:
                self.leaf_per_max_rank = self.config['params']['leaf_per_max_ranks']

        if self.scaling_type == 'weak':
            if self.leaf_per_rank is None:
                self.leaf_per_rank = self.config['params']['leaf_per_rank']

    def expand_omp(self, omp):
        """Expand omp sequence

        parameters
        ----------
        omp : [int]
        """
        if omp is None:
            max_threads = int(self.max_cores / 2)
            self.omp = tools.expand_power_sequence(largest=max_threads)

        elif isinstance(omp, int):
            self.omp = tools.expand_power_sequence(largest=omp)

        else:
            self.omp = np.array(omp)

    def expand_mpi(self, mpi):
        """Expand mpi sequences for each omp

        parameters
        ----------
        mpi : [int]
        """
        for omp in self.omp:
            max_ranks = int(self.max_cores / omp)

            if mpi is None:
                self.mpi[omp] = tools.expand_power_sequence(largest=max_ranks)
            elif isinstance(mpi, int):
                self.mpi[omp] = tools.expand_power_sequence(largest=mpi)
            else:
                self.mpi[omp] = np.array(mpi)

    def expand_leaf(self, leaf):
        """Expand leaf sequences for each omp

        parameters
        ----------
        leaf : [int]
        """
        for omp in self.omp:
            if leaf is None:
                if self.scaling_type == 'strong':
                    max_ranks = int(self.max_cores / omp)
                    self.leaf[omp] = max_ranks * np.array(self.leaf_per_max_rank)

                elif self.scaling_type == 'weak':
                    self.leaf[omp] = np.array(self.leaf_per_rank)
            else:
                self.leaf[omp] = tools.ensure_sequence(leaf)

    def load_models(self):
        """Load all model timing data
        """
        for omp in self.omp:
            self.models[omp] = {}

            for leaf in self.leaf[omp]:
                self.models[omp][leaf] = {}
                leaf_blocks = self.get_leaf_blocks(leaf=leaf, omp=omp)

                for i, mpi in enumerate(self.mpi[omp]):
                    print(f'\rLoading {omp}_{leaf_blocks[i]}_{mpi}', end=10*' ')

                    self.models[omp][leaf][mpi] = model.Model(
                                                        model_set=self.model_set,
                                                        omp=omp,
                                                        leaf_blocks=leaf_blocks[i],
                                                        mpi=mpi,
                                                        log_basename=self.log_basename,
                                                        which_table=self.which_table)
        print()

    def extract_data(self):
        """Extract multi-dimensional table of model timing data
        """
        print('Extracting performance data')
        omp_dict = {}

        for omp, omp_set in self.models.items():
            leaf_dict = {}

            for leaf, leaf_set in omp_set.items():
                mpi_dict = {}

                for mpi, mod in leaf_set.items():
                    mpi_dict[mpi] = mod.table.to_xarray()

                leaf_xr = xr.concat(mpi_dict.values(), dim='mpi')
                leaf_xr.coords['mpi'] = list(mpi_dict.keys())
                leaf_dict[leaf] = leaf_xr

            omp_xr = xr.concat(leaf_dict.values(), dim='leaf')
            omp_xr.coords['leaf'] = list(leaf_dict.keys())
            omp_dict[omp] = omp_xr

        full_xr = xr.concat(omp_dict.values(), dim='omp')
        full_xr.coords['omp'] = list(omp_dict.keys())

        self.data = full_xr
        self.extract_zupcs()

    def extract_zupcs(self):
        """Add ZUPCS to xarray table
        """
        if self.scaling_type == 'strong':
            leaf_blocks = self.data.leaf
        else:
            leaf_blocks = self.data.leaf * self.data.mpi

        zone_updates = self.n_timesteps * leaf_blocks * self.block_size**3
        core_seconds = self.data.omp * self.data.mpi * self.data['avg']

        zupcs = zone_updates / core_seconds
        self.data['zupcs'] = zupcs

    # =======================================================
    #                      Analysis
    # =======================================================
    def get_data(self, var, leaf, omp=None, mpi=None, unit=None):
        """Return variable from subset of performance data

        parameters
        ----------
        var : one of ['times', 'zupcs', 'speedup', 'efficiency']
        leaf : int
        omp : int
        mpi : int
        unit : str
        """
        function = {'times': self._get_times,
                    'zupcs': self._get_zupcs,
                    'speedup': self._get_speedup,
                    'efficiency': self._get_efficiency,
                    }.get(var)

        if function is None:
            raise ValueError(f"invalid var: '{var}'")

        return function(leaf=leaf, omp=omp, mpi=mpi, unit=unit)

    def _get_times(self, leaf, omp=None, mpi=None, unit=None):
        """Return array of runtimes versus mpi ranks

        parameters
        ----------
        leaf : int
        omp : int
        mpi : int
        unit : str
        """
        return self._slice_table(var=self.time_column, leaf=leaf,
                                 omp=omp, mpi=mpi, unit=unit)

    def _get_zupcs(self, leaf, omp=None, mpi=None, unit=None):
        """Return array of Zone Updates Per Core Second, versus mpi ranks

        parameters
        ----------
        leaf : int
        omp : int
        mpi : int
        unit : str
        """
        return self._slice_table(var='zupcs', leaf=leaf,
                                 omp=omp, mpi=mpi, unit=unit)

    def _get_speedup(self, leaf, omp=None, unit=None, mpi=None):
        """Return array of speedup versus MPI ranks

        parameters
        ----------
        leaf : int
        omp : int
        mpi : int
        unit : str
        """
        times = self._slice_table(var=self.time_column, leaf=leaf,
                                  omp=omp, mpi=mpi, unit=unit)
        return times[0] / times

    def _get_efficiency(self, leaf, omp=None, mpi=None, unit=None):
        """Return array of scaling efficiency versus MPI ranks

        parameters
        ----------
        leaf : int
        omp : int
        mpi : int
        unit : str
        """
        times = self._slice_table(var=self.time_column, leaf=leaf,
                                  omp=omp, mpi=mpi, unit=unit)

        if mpi is None:
            x = times.mpi
        else:
            x = times.omp

        eff_factor = {'strong': x,
                      'weak': 1.0
                      }.get(self.scaling_type)

        return 100 * times[0] / (eff_factor * times)

    def _slice_table(self, var, leaf, omp=None, mpi=None, unit=None):
        """Return slice of timing data versus mpi ranks or omp threads

        parameters
        ----------
        leaf : int
        var : str
        omp : int
        mpi : int
        unit : str
        """
        if var not in list(self.data.keys()):
            raise ValueError(f"var='{var}' not in table")
        if unit is None:
            unit = self.unit

        if mpi is None:
            if omp is None:
                raise ValueError("Must specify either 'omp' or 'mpi'")
            else:
                data = self.data.sel(omp=omp, leaf=leaf, unit=unit)[var]
                return data.dropna('mpi')

        elif omp is None:
            data = self.data.sel(mpi=mpi, leaf=leaf, unit=unit)[var]
            return data.dropna('omp')

        else:
            data = self.data.sel(mpi=mpi, leaf=leaf, unit=unit, omp=omp)[var]
            return data

    def get_model_table(self, leaf, omp, mpi):
        """Return timing table of specific model

        parameters
        ----------
        leaf : int
        omp : int
        mpi : int
        """
        m = self.models[omp][leaf][mpi]
        return m.table

    def get_leaf_blocks(self, leaf, omp):
        """Return array of total leaf blocks versus mpi

        parameters
        ----------
        leaf : int
        omp : int
        """
        if self.scaling_type == 'strong':
            return np.full_like(self.mpi[omp], leaf)

        elif self.scaling_type == 'weak':
            return leaf * self.mpi[omp]

    def get_leaf_array(self, x_var, omp, mpi):
        """Return array of leaf blocks versus either mpi or omp

        parameters
        ----------
        x_var : str
        omp : int
        mpi : int
        """
        self._check_x_var(x_var=x_var, omp=omp, mpi=mpi)
        if x_var == 'mpi':
            return self.mpi[omp]
        elif x_var == 'omp':
            # y =
            x = y.coords[x_var]

        pass

    # =======================================================
    #                      Plotting
    # =======================================================
    def plot_multiple(self, omp=None, y_vars=None,
                      unit=None, x_scale=None, y_scale=None,
                      sub_figsize=(5, 3)):
        """Plot multiple sets of models

        parameters
        ----------
        omp : int or [int]
        y_vars : [str]
            types of plots
        unit : str
        x_scale : str
        y_scale : str
        sub_figsize : [width, height]
        """
        if omp is None:
            omp = self.omp
        else:
            omp = tools.expand_sequence(omp)

        if y_vars is None:
            y_vars = self.config['plot']['multiplot'][self.scaling_type]

        nrows = len(omp)
        ncols = len(y_vars)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(sub_figsize[0]*ncols, sub_figsize[1]*nrows))

        for i, omp_threads in enumerate(omp):
            for j, y_var in enumerate(y_vars):
                ax = axes[i, j]
                self.plot(x_var='mpi', y_var=y_var, omp=omp_threads, ax=ax,
                          unit=unit, data_only=True)

                self._set_ax_subplot(axes=axes, row=i, col=j, omp=omp_threads,
                                     x_var='mpi', y_var=y_var,
                                     x_scale=x_scale, y_scale=y_scale)
        plt.tight_layout()
        return fig

    def plot(self, x_var, y_var, omp=None, mpi=None, unit=None, x_scale=None,
             ax=None, data_only=False, marker='o', linestyle='-'):
        """Plot performance scaling versus OMP threads or MPI ranks

        parameters
        ----------
        x_var : 'omp' or 'mpi'
        y_var : one of ['times', 'zupcs', 'speedup', 'efficiency']
        omp : int
        mpi : int
        unit : str
        x_scale : str
        ax : Axis
        data_only : bool
        marker : str
        linestyle : str
        """
        fig, ax = self._setup_fig_ax(ax=ax)
        self._check_x_var(x_var=x_var, omp=omp, mpi=mpi)
        max_x = 1

        for leaf in self.leaf[omp]:
            label = leaf
            # label = {False: int(leaf)}.get(data_only)

            y = self.get_data(var=y_var, leaf=leaf, omp=omp, mpi=mpi, unit=unit)
            x = y.coords[x_var]
            max_x = max(max_x, x[-1])
            ax.plot(x, y, label=label, marker=marker, linestyle=linestyle)

        self._set_ax(ax=ax, x_var=x_var, y_var=y_var, omp=omp,
                     x_scale=x_scale, data_only=data_only)

        return fig

    def plot_omp(self, mpi, y_var, unit=None, x_scale=None,
                 ax=None, data_only=False,
                 marker='o', linestyle='-'):
        """Plot OMP thread scaling
        """
        if unit is None:
            unit = self.unit

        fig, ax = self._setup_fig_ax(ax=ax)

        data = self.data.sel(mpi=mpi, unit=unit)[self.time_column]

        for leaf in data.leaf:
            label = {False: int(leaf)}.get(data_only)

            y = self.get_data(var=y_var, leaf=leaf, mpi=mpi, unit=unit)
            x = y.omp
            ax.plot(x, y, marker=marker, linestyle=linestyle, label=label)

        self._set_ax(ax=ax, x_var='omp', y_var=y_var, omp=mpi,
                     x_scale=x_scale, data_only=data_only)

        return fig

    # =======================================================
    #                      Plot tools
    # =======================================================
    def _setup_fig_ax(self, ax):
        """Setup fig, ax, checking if ax already provided

        parameters
        ----------
        ax : Axis
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        return fig, ax

    def _check_x_var(self, x_var, omp, mpi):
        """Check for valid x_var and args

        parameters
        ----------
        x_var : str
        omp : int
        mpi : int
        """
        x_map = {'omp': mpi, 'mpi': omp}
        name_map = {'omp': 'mpi', 'mpi': 'omp'}

        if x_map[x_var] is None:
            raise ValueError(f"must specify {name_map[x_var]} if x_var='{x_var}'")

    def _set_ax(self, ax, x_var, y_var, omp,
                x_scale=None, y_scale=None, data_only=False):
        """Set axis properties

        parameters
        ----------
        ax : Axis
        x_var : str
        y_var : str
        omp : int
        x_scale : str
        y_scale : str
        data_only : bool
        """
        fixed_map = {'omp': 'mpi', 'mpi': 'omp'}

        if not data_only:
            self._set_ax_legend(ax=ax)
            self._set_ax_title(ax=ax)
            self._set_ax_labels(ax=ax, x_var=x_var, y_var=y_var)
            self._set_ax_scale(ax=ax, x_var=x_var, y_var=y_var,
                               x_scale=x_scale, y_scale=y_scale)
            self._set_ax_xticks(ax=ax)
            self._set_ax_dashed(ax=ax, y_var=y_var)
            self._set_ax_text(ax=ax, omp=omp, fixed_var=fixed_map[x_var])

    def _set_ax_subplot(self, axes, x_var, y_var, row, col, omp,
                        x_scale, y_scale):
        """Set axis properties for subplot (see plot_multiple)

        parameters
        ----------
        axes : Axes
        x_var : str
        y_var : str
        row : int
        col : int
        omp : int
        x_scale : str
        y_scale : str
        """
        ax = axes[row, col]
        nrows = axes.shape[0]
        ncols = axes.shape[1]

        if col == 0:
            self._set_ax_text(ax=ax, omp=omp, fixed_var='omp')
            if self.scaling_type == 'strong':
                self._set_ax_legend(ax=ax)

            if row == 0:
                self._set_ax_title(ax=ax)
                if self.scaling_type == 'weak':
                    self._set_ax_legend(ax=ax)

        if row == nrows - 1:
            ax.set_xlabel(self.config['plot']['labels'][x_var])

        ax.set_ylabel(self.config['plot']['labels'][y_var])

        self._set_ax_scale(ax=ax, x_var=x_var, y_var=y_var,
                           x_scale=x_scale, y_scale=y_scale)
        self._set_ax_xticks(ax=ax)
        self._set_ax_dashed(ax=ax, y_var=y_var)

    def _set_ax_legend(self, ax):
        """Set axis legend

        parameters
        ----------
        ax : Axis
        """
        titles = {'strong': 'Leaf blocks',
                  'weak': 'Leaf blocks / rank'}
        ax.legend(title=titles[self.scaling_type])

    def _set_ax_title(self, ax):
        """Set axis title

        parameters
        ----------
        ax : Axis
        """
        ax.set_title(f'{self.model_set}, unit={self.unit}')

    def _set_ax_text(self, ax, omp, fixed_var):
        """Set axis text

        parameters
        ----------
        ax : Axis
        omp : int
        fixed_var : str
            which of omp/mpi are held fixed
        """
        label = self.config['plot']['labels'].get(fixed_var, fixed_var)
        ax.text(0.5, 0.95, f'{label} = {omp}',
                verticalalignment='center', horizontalalignment='center',
                fontsize=12, transform=ax.transAxes)

    def _set_ax_labels(self, ax, x_var, y_var):
        """Set axis labels

        parameters
        ----------
        ax : Axis
        x_var : str
        y_var : str
        """
        def get_label(key):
            return self.config['plot']['labels'].get(key, key)

        ax.set_xlabel(get_label(x_var))
        ax.set_ylabel(get_label(y_var))

    def _set_ax_xticks(self, ax):
        """Set axis ticks

        parameters
        ----------
        ax : Axis
        """
        x = tools.expand_power_sequence(largest=self.max_cores)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks(x)
        ax.tick_params(axis='x', which='minor', bottom=False)

    def _set_ax_dashed(self, ax, y_var):
        """Plot dashed line on axis
        """
        x = [1, self.max_cores]
        if y_var == 'efficiency':
            ax.plot(x, [100, 100], ls='--', color='black')
        elif y_var == 'speedup':
            ax.plot(x, x, ls='--', color='black')

    def _set_ax_scale(self, ax, x_var, y_var, x_scale, y_scale):
        """Set axis scales

        parameters
        ----------
        ax : Axis
        x_var : str
        y_var : str
        x_scale : str
        y_scale : str
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

