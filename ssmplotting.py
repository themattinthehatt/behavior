import os
import numpy as np
import matplotlib.pyplot as plt


def plot_dynamics_matrices(model, deridge=False, tag=None):
    K = model.K
    n_lags = model.observations.lags
    if n_lags == 1:
        n_cols = 3
        fac = 1
    elif n_lags == 2:
        n_cols = 3
        fac = 1 / n_lags
    elif n_lags == 3:
        n_cols = 3
        fac = 1.25 / n_lags
    elif n_lags == 4:
        n_cols = 3
        fac = 1.50 / n_lags
    elif n_lags == 5:
        n_cols = 2
        fac = 1.75 / n_lags
    else:
        n_cols = 1
        fac = 1
    n_rows = int(np.ceil(K / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows * fac))

    if str(model.observations.__class__).find('hierarchical') > -1:
        if tag is None:
            mats = np.copy(model.observations.global_ar_model.As)
        else:
            mats = np.copy(model.observations.get_As(tag))
    else:
        mats = np.copy(model.observations.As)
    if deridge:
        for k in range(K):
            for l in range(model.observations.lags):
                for d in range(model.D):
                    mats[k, d, model.D*l + d] = np.nan
        clim = np.nanmax(np.abs(mats))
    else:
        clim = np.max(np.abs(mats))

    for k in range(K):
        plt.subplot(n_rows, n_cols, k + 1)
        im = plt.imshow(mats[k], cmap='RdBu_r', clim=[-clim, clim])
        plt.xticks([])
        plt.yticks([])
        plt.title('State %i' % k)
    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    return fig


def plot_biases(model, tag=None):
    fig = plt.figure(figsize=(6, 4))

    if str(model.observations.__class__).find('hierarchical') > -1:
        if tag is None:
            mats = np.copy(model.observations.global_ar_model.bs.T)
        else:
            mats = np.copy(model.observations.get_bs(tag).T)
    else:
        mats = np.copy(model.observations.bs.T)

    clim = np.max(np.abs(mats))
    im = plt.imshow(mats, cmap='RdBu_r', clim=[-clim, clim], aspect='auto')
    plt.xlabel('State')
    plt.yticks([])
    plt.ylabel('Observation dimension')
    plt.tight_layout()
    plt.colorbar()
    plt.title('State biases')
    plt.show()
    return fig


def plot_state_transition_matrix(model, deridge=False, tag=None):

    if str(model.transitions.__class__).find('hierarchical') > -1:
        if tag is None:
            raise NotImplementedError
        else:
            trans = np.copy(model.transitions.get_transition_matrix(tag))
    else:
        trans = np.copy(model.transitions.transition_matrix)

    if deridge:
        n_states = trans.shape[0]
        for i in range(n_states):
            trans[i, i] = np.nan
        clim = np.nanmax(np.abs(trans))
    else:
        clim = 1
    fig = plt.figure()
    plt.imshow(trans, clim=[-clim, clim], cmap='RdBu_r')
    plt.colorbar()
    plt.title('State transition matrix')
    plt.show()
    return fig


def plot_covariance_matrices(model, tag=None):
    K = model.K
    n_cols = 3
    n_rows = int(np.ceil(K / n_cols))

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    if str(model.observations.__class__).find('hierarchical') > -1:
        if tag is None:
            mats = np.copy(model.observations.global_ar_model.Sigmas)
        else:
            mats = np.copy(model.observations.get_Sigmas(tag))
    else:
        mats = np.copy(model.observations.Sigmas)

    clim = np.quantile(np.abs(mats), 0.95)

    for k in range(K):
        plt.subplot(n_rows, n_cols, k + 1)
        im = plt.imshow(mats[k], cmap='RdBu_r', clim=[-clim, clim])
        plt.xticks([])
        plt.yticks([])
        plt.title('State %i' % k)
    plt.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.4, 0.03, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    return fig
