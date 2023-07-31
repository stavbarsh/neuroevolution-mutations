import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator


class Objective:
    def __init__(self):
        self.avg = 0
        self.max = 0


class Result:
    def __init__(self):
        self.gen = 0
        self.fit = 0
        self.dist = 0
        self.rew = 0
        self.generality = 0
        self.mu = 0
        self.std = 0
        self.skew = 0
        self.kurt = 0
        self.mode = 0
        self.obj0 = Objective()
        self.obj1 = Objective()


def get_value(line: str):
    return float(line.split(':')[3][:-1])


def get_name(line: str):
    try:
        val = line.split(':')[2]
    except:
        val = None
    return val


def graph(file: str, plotlast=0, n_avg=10, view=False, filename='plot_graph', filepath=None, fig_n=10):
    results = []
    with open(file) as f:
        for line in f.readlines():
            val = get_name(line)
            if val == 'gen':
                result = Result()
                result.gen = get_value(line)
                results.append(result)
            elif val == 'best':
                pass
            elif val == 'obj 0 avg':
                result.obj0.avg = get_value(line)
            elif val == 'obj 0 max':
                result.obj0.max = get_value(line)
            elif val == 'obj 1 avg':
                result.obj1.avg = get_value(line)
            elif val == 'obj 1 max':
                result.obj1.max = get_value(line)
            elif val == 'fit':
                result.fit = get_value(line)
            elif val == 'dist':
                result.dist = get_value(line)
            elif val == 'rew':
                result.rew = get_value(line)
            elif val == 'generality':
                result.generality = get_value(line)
            elif val == 'mu':
                result.mu = get_value(line)
            elif val == 'std':
                result.std = get_value(line)
            elif val == 'skew':
                result.skew = get_value(line)
            elif val == 'kurt':
                result.kurt = get_value(line)
            elif val == 'mode':
                result.mode = get_value(line)

    # remove last result if empty
    if len(results) > 1:
        if results[-1].fit == 0:
            results = results[:-1]

        l = plotlast if (plotlast > 0) and (len(results) >= plotlast) else len(results)
        gens = [r.gen for r in results[-l:]]
        n_avg = 1 if gens[-1] <= n_avg else n_avg
        kernel = np.ones(n_avg) / n_avg

        i = 0
        # plot results for the hole evolution process
        plt.figure(fig_n + i)
        plt.plot(gens, [r.fit for r in results[-l:]], label=f'result')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.dist for r in results[-l:]], label=f'dist')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.rew for r in results[-l:]], label=f'reward')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.generality for r in results[-l:]], label=f'generality')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        max_obj0 = [r.obj0.max for r in results[-l:]]
        avg_obj0 = [r.obj0.avg for r in results[-l:]]
        sm_max_obj0 = np.convolve(max_obj0, kernel, mode='valid')
        sm_avg_obj0 = np.convolve(avg_obj0, kernel, mode='valid')
        p = plt.plot(gens[n_avg - 1:], sm_avg_obj0, label=f'avg res')
        plt.plot(gens, avg_obj0, color=p[0].get_color(), alpha=0.3)
        p = plt.plot(gens[n_avg - 1:], sm_max_obj0, label=f'max res')
        plt.plot(gens, max_obj0, color=p[0].get_color(), alpha=0.3)
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        max_obj1 = [r.obj1.max for r in results[-l:]]
        avg_obj1 = [r.obj1.avg for r in results[-l:]]
        sm_max_obj1 = np.convolve(max_obj1, kernel, mode='valid')
        sm_avg_obj1 = np.convolve(avg_obj1, kernel, mode='valid')
        p = plt.plot(gens[n_avg - 1:], sm_avg_obj1, label=f'avg nov')
        plt.plot(gens, avg_obj1, color=p[0].get_color(), alpha=0.3)
        p = plt.plot(gens[n_avg - 1:], sm_max_obj1, label=f'max nov')
        plt.plot(gens, max_obj1, color=p[0].get_color(), alpha=0.3)
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.mu for r in results[-l:]], label=f'policy mu')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.std for r in results[-l:]], label=f'policy std')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.skew for r in results[-l:]], label=f'policy skew')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.kurt for r in results[-l:]], label=f'policy kurt')
        plt.legend()
        i += 1

        plt.figure(fig_n + i)
        plt.plot(gens, [r.mode for r in results[-l:]], label=f'policy mode')
        plt.legend()
        i += 1

        if filepath is not None:
            os.makedirs(filepath, exist_ok=True)
            for i in range(fig_n, fig_n + i):
                plt.figure(i)
                plt.savefig(filepath + "/" + filename + "-" + str(i))

        if view:
            plt.show()

        plt.close('all')

    return results


def draw_gen(value, gen, color_idx=0, init=True, view=True, filename=None, filepath=None, fig_n=0, ylim=300):
    if init:
        fig = plt.figure(fig_n)
        ax = fig.subplots()
        fig.set_dpi(100)
        ax.set_ylim(0, ylim)
    else:
        plt.figure(fig_n)
        ax = plt.gca()

    circle = plt.Circle((gen, value), ylim / 300, facecolor=plt.cm.tab20(color_idx))
    ax.add_patch(circle)
    ax.set_xlim(0, gen + 2)

    if filename is not None:
        os.makedirs(filepath, exist_ok=True)
        plt.savefig(filepath + "/" + filename + "-" + str(fig_n))

    if view:
        plt.show()
