"""
Plots for TFBS aka the ground truth data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_tfbs_by_position(tfbs: np.ndarray, savepath: str | None = None):

    individual = tfbs[:, :, :57]
    aggregated = tfbs[:, :, 57]

    # Look at binding sites per position
    positions = np.arange(1003)
    individual_by_pos = individual.sum(axis=-1).sum(axis=0)
    aggregated_by_pos = aggregated.sum(axis=0)

    xlabel = "Position"
    ylabel = "No. TFBS"
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    ax[0].bar(positions + 1, individual_by_pos)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("Individual TFs summed up")
    ax[1].bar(positions + 1, aggregated_by_pos)
    ax[1].set_xlabel(xlabel)
    ax[1].set_title("Aggregated track")

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


def plot_tfbs_by_tf(tfbs: np.ndarray, savepath: str | None = None):
    individual = tfbs[:, :, :57]

    # Look at binding sites by TF
    individual_by_tf = individual.sum(axis=0).sum(axis=0)
    tfs = np.arange(1, 58)
    plt.bar(tfs, individual_by_tf)
    plt.xlabel("TF")
    plt.ylabel("No. TFBS")

    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


def plot_tfbs_by_seq(tfbs: np.ndarray, savepath: str | None = None):
    individual = tfbs[:, :, :57]
    aggregated = tfbs[:, :, 57]

    # Look at #binding sites over sequences distribution
    individual_by_seq = individual.sum(axis=-1).sum(axis=-1)
    aggregated_by_seq = aggregated.sum(axis=-1)

    ylabel = "No. TFBS"
    plt.boxplot([individual_by_seq, aggregated_by_seq])
    plt.xticks(
        ticks=[1, 2], labels=["Individual TFs summed up", "Aggregated track"]
    )
    plt.ylabel(ylabel)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    tfbs_filepath = os.path.join(
        data_dir, "chip_exo_57_TF_binding_sites_30subset.npy"
    )
    tfbs = np.load(tfbs_filepath)

    plot_tfbs_by_position(tfbs)

    plot_tfbs_by_tf(tfbs)

    plot_tfbs_by_seq(tfbs)
