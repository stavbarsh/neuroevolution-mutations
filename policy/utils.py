import os
import math
import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def plot_param_dist(params, bins=1000, density=True, filename='param_dist', filepath=None, fig_n=0, view=False):
    plt.figure(fig_n)
    _, (min_w, max_w), mean, var, skew, kurt = stats.describe(params)
    label = f'{filename} - params: {len(params) / 1e3:.2f} [K], mean: {mean:.4f}, var: {var:.4f}, skew: {skew:.4f}, kurt: {kurt:.4f}'
    # plt.hist(params, bins, density=density, label=label)
    count, bins = np.histogram(params, bins, density=density)
    plt.plot(bins[:-1], count, label=label)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25))
    lim = np.max(np.abs([min_w, max_w])) * 1.1
    plt.xlim(-lim, lim)
    plt.tight_layout()
    if filepath is not None:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(filepath + "/" + filename + "-" + str(fig_n))
    if view:
        plt.show()
    plt.close()
    return count, bins


def get_flat(module: torch.nn.Module) -> np.ndarray:
    return torch.cat([t.flatten() for t in module.state_dict().values()]).numpy()
