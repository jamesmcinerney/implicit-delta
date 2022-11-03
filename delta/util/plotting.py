import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def plot_fn_eb(query_xs, mn, sd, ylims=None, data=None, fpath=None):
    plt.figure(figsize=(3, 4))  # (3,4)
    x_min_plt = query_xs.min()
    x_max_plt = query_xs.max()
    if data is not None:
        X, Y = data
        plt.scatter(X, Y, label='data', marker='x')
    plt.xlim(x_min_plt, x_max_plt)
    if ylims is not None:
        y_min_plt, y_max_plt = ylims
        plt.ylim(y_min_plt, y_max_plt)
    plt.plot(query_xs, mn, color='red', label='mean')
    plt.plot(query_xs, (mn + 1.96 * sd), color='green', ls='--', label='95% C.I.')
    plt.plot(query_xs, (mn - 1.96 * sd), color='green', ls='--')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    if fpath is not None:
        plt.savefig(fpath, dpi=30, bbox_inches="tight")


def plot_cov(C, fpath=None, vmin=None, vmax=None):
    plt.figure(figsize=(3, 4))  # (3,4)
    hshow = plt.imshow(C, aspect='auto', vmin=vmin, vmax=vmax)  # plt.matshow(C)
    plt.xlabel('prediction')
    plt.ylabel('prediction')
    cbar = plt.colorbar(hshow, ax=plt.gca(), shrink=0.8)  # , extend='both')
    cbar.minorticks_on()
    if fpath is not None:
        plt.savefig(fpath, dpi=30, bbox_inches="tight")


