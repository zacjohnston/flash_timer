import matplotlib.pyplot as plt

from . import model_set


def load_omp(mpi=1,
             omp=64,
             leaf=(128, 256, 512),
             unit='Hydro',
             ):
    models = {}

    for mset in ['spark', 'uhd']:
        models[mset] = model_set.ModelSet(scaling_type='strong',
                                          model_set=f'sod3d_flashx_{mset}_strong_omp',
                                          mpi=mpi,
                                          omp=omp,
                                          leaf=leaf,
                                          unit=unit)

    return models


def plot_omp(models, mpi=1, y_var='times'):
    fig = models['spark'].plot_omp(mpi=mpi, y_var=y_var)
    plt.gca().set_prop_cycle(None)

    models['uhd'].plot_omp(mpi=mpi, y_var=y_var, marker='x', linestyle='--',
                           ax=fig.axes[0], data_only=True)

    fig.axes[0].plot([8, 8], [100, 100], ls='--', label='UHD', color='black')
    models['spark']._set_ax_legend(ax=fig.axes[0])
    return fig
