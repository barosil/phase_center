# -*- coding: utf-8 -*-
"""Utilidades para lidar com feixes de antenas."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from lmfit import Parameters, minimize
from scipy.constants import c as c
from scipy.integrate import quad_vec

# import scipy.integrate
# import scipy.optimize
# import scipy.interpolate
from scipy.special import j0


def plot_beams(
    data=None, field="AMPLITUDE", freqs=None, limits=None, ax=None, **kwargs
):
    """Visualização de feixes.

    Args:
        data (pd.DataFrame): dataset `data`. Defaults to None.
        field (str): Campo utilizado para visualização: `AMPLITUDE` ou `PHASE`. Defaults to `AMPLITUDE`.
        freqs (list): lista de frequências para o gráfico. Default None significa todas.
        limits (list): limits x e y para o gráfico. Defaults to None.
        ax (ax): matplotlib `ax`. Defaults to None cria uma nova figura.
        **kwargs (type): `**kwargs` são pssados diretamente para matplotib.

    Returns:
        type: ax.

    """
    if freqs:
        data = data[data.FREQ.isin(freqs)]
    for freq, beam in data.groupby("FREQ"):
        if ax is None:
            fig, ax = plt.subplots()
        plt.plot(beam["ANGLE"], beam[field], label=r"{:.3f} GHz".format(freq), **kwargs)
        if limits:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
    ax.set_xlabel(r"$\theta (^\circ)$")
    ax.set_ylabel(field)
    ax.legend(loc="lower center", ncol=5)
    ax.grid(color="lightgray", linewidth=0.3, axis="y")

    return ax


def _interpolate_beam(x, data):
    """Função auxiliar para obter máximos. É uma interpolação simples com sinal negativo."""
    result = sp.interpolate.interp1d(data.ANGLE, data.AMPLITUDE)
    return -result(x)


def normalize(data=None):
    """Centraliza dados da amplitude dos feixes e normaliza todos para terem máximo em zero."""
    data = data[["FREQ", "ANGLE", "AMPLITUDE", "PHASE"]]
    dfs = []
    for freq, df in data.groupby("FREQ"):
        maximum = sp.optimize.minimize(
            _interpolate_beam, 0, method="nelder-mead", args=df
        )
        theta_max = maximum.x
        offset = maximum.fun
        result = pd.DataFrame(columns=["FREQ", "ANGLE", "AMPLITUDE", "PHASE"])
        result["ANGLE"] = df["ANGLE"] - theta_max
        result["AMPLITUDE"] = df["AMPLITUDE"] + offset
        result["FREQ"] = freq
        result["PHASE"] = df["PHASE"]
        dfs.append(result)
    result = pd.concat(dfs)
    return result


def fit_taper(data=None, level=-3, x0=None):
    """Determina valor de taper do dataframe `data` no nível `level` usando como valor inicial `x0`."""
    taper_func = lambda theta: _interpolate_beam(theta, data) + level
    taper = sp.optimize.root(taper_func, x0, method="lm")
    return taper


def get_tapers(data=None, level=-3, x0=None):
    """Determina os valores de taper no nível `level` para todas as frequências presentes o dataframe `data` usando valor inicial `x0`."""
    taper = []
    err = []
    freqs = []
    for freq, df in data.groupby("FREQ"):
        res = fit_taper(data=df, level=level, x0=x0)
        taper.append(res.x[0])
        err.append(res.cov_x[0][0])
        freqs.append(freq)
    result = pd.DataFrame(columns=["FREQ", "TAPER", "ERR"])
    result.FREQ = freqs
    result.TAPER = taper
    result.ERR = err
    return result


def EE_gauss(rr, ww):
    result = np.exp(-(rr**2) / ww**2)
    return result


def B_model(theta, freq, ww, aa):
    uu = 2 * np.pi * freq * aa * np.sin(np.radians(theta)) / c
    _num = lambda rr: EE_gauss(aa * rr, ww) * j0(uu * rr) * rr
    _den = lambda rr: EE_gauss(aa * rr, ww) * rr

    num = quad_vec(_num, 0, 1)[0]
    den = quad_vec(_den, 0, 1)[0]

    result = np.abs(num**2 / den**2)

    return result


def residual(params, theta, data, weigths):
    ww = params["w"]
    aa = params["A"]
    freq = params["freq"]
    model = B_model(theta, freq, ww, aa)
    result = np.abs(data**2 - model**2) / (weigths**2)
    return result


def fit_beam(data=None, weigths=None, freq=None, theta_max=30, ww=100, aa=10, power=2):
    theta_range = theta_max
    params = Parameters()
    params.add("w", value=ww, min=0)
    params.add("A", value=aa, min=0)
    params.add("freq", value=freq * 1e9, vary=False)
    df = data.query("ANGLE < @theta_range and ANGLE > - @theta_range and FREQ==@freq")
    thetas = df.ANGLE
    dados = 10 ** (df.AMPLITUDE / 10)
    if not weigths:
        weigths = np.ones_like(thetas)
    out = minimize(residual, params, args=(thetas, dados, weigths))

    return out
