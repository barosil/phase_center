import lmfit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c


def get_freqs(mode="V", path="../data/raw/"):
    dataset_H = path + "beampattern_horn01_Polarização_Horizontal_Copolar.csv"
    dataset_V = path + "beampattern_horn01_Polarização_Vertical_Copolar.csv"
    if mode == "V":
        data = pd.read_csv(dataset_V)
    else:
        data = pd.read_csv(dataset_H)
    return data.FREQ.unique()


def normalize_AMP(data):
    df = data.copy()
    df["AMPLITUDE"] = data.groupby("FREQ")[["AMPLITUDE"]].transform(
        lambda x: x - x.max()
    )
    return df


def cut_theta(data, theta_max_deg):
    theta_max = np.radians(theta_max_deg)
    df = (
        data.query("ANGLE >= -@theta_max & ANGLE <= @theta_max")
        .copy()
        .reset_index(drop=True)
    )
    return df


def get_max_amp(data):
    theta_0 = data.ANGLE.iloc[data.AMPLITUDE.idxmax()]
    return theta_0


def normalize_PHASE(data, angle_correction=True, wrap=True):
    data = data.copy()
    if angle_correction:
        theta_0 = get_max_amp(data)
        PHI_0 = data.PHASE.iloc[data.AMPLITUDE.idxmax()]
        data["ANGLE"] = data.ANGLE - theta_0
    else:
        PHI_0 = data.PHASE.iloc[data.ANGLE.abs().idxmin()]
        theta_0 = 0

    data["PHASE"] = data.PHASE - PHI_0
    data["delta_theta"] = theta_0
    if not wrap:
        data["PHASE"] = np.unwrap(data.PHASE)
    return data


def get_data(
    freq=1.0, mode="V", theta_max_deg=None, angle_correction=True, path="../data/raw/", wrap=True
):
    dataset_H = path + "beampattern_horn01_Polarização_Horizontal_Copolar.csv"
    dataset_V = path + "beampattern_horn01_Polarização_Vertical_Copolar.csv"
    if mode == "V":
        data = pd.read_csv(dataset_V)
    else:
        data = pd.read_csv(dataset_H)
    data = data.query("FREQ==@freq").copy().reset_index(drop=True)
    data.ANGLE = np.radians(data.ANGLE)
    data.PHASE = np.radians(data.PHASE)
    data.PHASE = (data.PHASE.values + data.PHASE.values[::-1]) / 2
    # Normalize amplitudes such that the maximum amplitude is 0 dB.
    data = normalize_AMP(data)
    # Normalize phases such that the phase at the maximum amplitude is 0.
    data = normalize_PHASE(data, angle_correction=angle_correction, wrap=wrap)
    if theta_max_deg is not None:
        data = cut_theta(data, theta_max_deg)
    return data


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    result = (theta + np.pi) % (2 * np.pi) - np.pi
    return result


def k0(freq: float) -> float:
    k0 = 2 * np.pi * freq * 1e9 / c
    return k0


def weight_uniform(data) -> np.ndarray:
    return np.ones(data.size)


def weight_Amp(data) -> np.ndarray:
    return 10 ** (data / 10)


def model(DXY: float, DZ: float, theta, wrap=True) -> np.ndarray:
    if wrap:
        Phi = wrap_angle(DXY * np.sin(theta) + DZ * np.cos(theta))
    else:
        Phi = DXY * np.sin(theta) + DZ * np.cos(theta)
    return Phi


def residuals(DXY, DZ, theta, Phase, weights, wrap=True, mode="V") -> np.ndarray:
    if mode == "H":
        Phase = Phase + np.pi / 2
    NORM = 1 / np.sum(weights)
    if wrap:
        res = (1 / 2) * NORM * weights * np.abs((wrap_angle(Phase - model(DXY, DZ, theta, wrap=wrap))) ** 2)
    else:
        res = (1 / 2) * NORM * weights * np.abs((Phase - model(DXY, DZ, theta, wrap=wrap)) ** 2)
    return res


def res2fit(params, theta, data):
    DXY = params["DXY"]
    DZ = params["DZ"]
    model_ = model(DXY, DZ, theta)
    res = model_ - data
    return res


def chi_2(DXY, DZ, theta, Phase, weights, mode="V"):
    res = np.sum(residuals(DXY, DZ, theta, Phase, weights, mode=mode), axis=1)
    return res


def _chi_2(DXY, DZ, data, weight_func, wrap=True, mode="V"):
    theta = data.ANGLE.values
    Phase = data.PHASE.values
    weights = weight_func(data.AMPLITUDE.values)
    res = np.sum(residuals(DXY, DZ, theta, Phase, weights, wrap=wrap, mode=mode), axis=1)
    return res


def fit_phase_center(DZ, theta, Phase, weights, wrap=True, mode="V"):
    params = lmfit.Parameters()
    params.add("DXY", value=0.0)
    params.add("DZ", value=DZ)

    # Cost function in lmfit format
    def res(params):
        params = params.valuesdict()
        DXY = params["DXY"]
        DZ = params["DZ"]
        return residuals(DXY, DZ=DZ, theta=theta, Phase=Phase, weights=weights, wrap=wrap, mode=mode)

    # Minimization
    result = lmfit.minimize(res, params)

    return result


def chi_2_set(Z_min, Z_max, theta_max_deg, n_points=1000, angle_correction=True, wrap=True):
    freqs = get_freqs()
    result = np.zeros((2, 2, len(freqs), n_points))
    fitter = np.zeros((2, 2, len(freqs)), dtype=object)
    for ii, mode in enumerate(["V", "H"]):
        for jj, weight_func in enumerate([weight_Amp, weight_uniform]):
            for kk, freq in enumerate(freqs):
                data = get_data(
                    freq,
                    mode=mode,
                    theta_max_deg=theta_max_deg,
                    angle_correction=angle_correction, wrap=wrap
                )

                DZ = np.linspace(Z_min, Z_max, n_points).reshape(-1, 1)
                DXY = 0.0
                chi_2 = _chi_2(DXY, DZ, data, weight_func, wrap=wrap, mode=mode)
                result[ii, jj, kk] = chi_2

                idx = np.argmin(chi_2)
                DZ_0 = 1
                theta = data.ANGLE.values
                Phase = data.PHASE.values
                weights = weight_func(data.AMPLITUDE.values)
                fit_ = fit_phase_center(DZ_0, theta, Phase, weights, wrap=wrap, mode=mode)
                fitter[ii, jj, kk] = fit_

    return result, fitter


def plot_chi2_set(data, n_freqs=1, Z_min=0, Z_max=15 * np.pi, ax=None, title=None):
    freqs = get_freqs()
    idxs = np.arange(len(freqs))[:: int(np.ceil(len(freqs) / n_freqs))]
    if not ax:
        fig, ax = plt.subplots(
            ncols=data.shape[0], nrows=data.shape[1], figsize=(12, 10)
        )
    if title:
        fig.suptitle(title, fontsize=16)
    cmap = mpl.colormaps["viridis"].resampled(data.shape[2])
    modes = ["V", "H"]
    weight_str = ["Amp", "Uniform"]
    for ii, res_modes in enumerate(data):
        for jj, res_weigths in enumerate(res_modes):
            for kk in idxs:
                chi_2 = res_weigths[kk]
                DZ = np.linspace(Z_min, Z_max, chi_2.shape[0])
                k_0 = k0(freqs[kk])
                ax[ii, jj].plot(DZ / k_0, chi_2, label=f"{freqs[kk]}", color=cmap(kk))
                ax[ii, jj].set_title(
                    f"mode = {modes[ii]}, weights = {weight_str[jj]}", fontsize=8
                )
                ax[ii, jj].set_xlabel(
                    r"$\Delta Z / k_0$ (Physical distance (m))", fontsize=8
                )
                ax[ii, jj].set_ylabel(r"$\chi^2$")
    ax[1, 0].legend(
        loc="lower center", bbox_to_anchor=(1.1, -0.3), fontsize=8, ncols=10
    )

    return ax


def plot_fit_set(fitter, Z_min=0, Z_max=15 * np.pi, ax=None, title=None):
    freqs = get_freqs()
    if not ax:
        fig, ax = plt.subplots(
            ncols=fitter.shape[0], nrows=fitter.shape[1], figsize=(12, 10)
        )
    if title:
        fig.suptitle(title, fontsize=16)
    cmap = mpl.colormaps["viridis"].resampled(2)
    modes = ["V", "H"]
    weight_str = ["Amp", "Uniform"]
    for ii, res_modes in enumerate(fitter):
        for jj, res_weigths in enumerate(res_modes):
            DXY = np.zeros_like(freqs)
            DXYerr = np.zeros_like(freqs)
            DZ = np.zeros_like(freqs)
            DZerr = np.zeros_like(freqs)
            for kk, fit_ in enumerate(res_weigths):
                k_0 = k0(freqs[kk])
                DXY[kk] = 100 * fit_.params["DXY"].value / k_0
                DXYerr[kk] = 100 * fit_.params["DXY"].stderr / k_0
                DZ[kk] = 100 * fit_.params["DZ"].value / k_0
                DZerr[kk] = 100 * fit_.params["DZ"].stderr / k_0

            ax[0, jj].errorbar(
                freqs, DZ, DZerr, fmt=".", color=cmap(ii), label=modes[ii]
            )
            ax[1, jj].errorbar(
                freqs, DXY, DXYerr, fmt=".", color=cmap(ii), label=modes[ii]
            )
            ax[0, jj].set_title(f"weights = {weight_str[jj]}", fontsize=8)
            ax[1, jj].set_title(f"weights = {weight_str[jj]}", fontsize=8)
            ax[0, jj].set_xlabel(r"Frequency (GHz)", fontsize=8)
            ax[0, jj].set_ylabel(
                r"$\Delta_Z / k_0$ (Physical distance (cm))r", fontsize=8
            )
            ax[1, jj].set_ylabel(
                r"$\Delta_{XY} / k_0$ (Physical distance (cm))r", fontsize=8
            )
            ax[0, jj].legend(fontsize=8)
            ax[1, jj].legend(fontsize=8)
        # )

    return ax


def plot_fit_combined(
    fitters, ax=None, title=None, weight=0, D0=0, legend=["15", "45", "90", "180"], **kwargs
):
    freqs = get_freqs()
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 5))
    if title:
        fig.suptitle(title, fontsize=16)
    cmap = mpl.colormaps["viridis"].resampled(2)
    modes = ["V", "H"]
    weight_str = ["Amp", "Uniform"]
    mm = ["o", "x", "s", "d"]
    for jj, fitter in enumerate(fitters):
        fitter = fitter[:, weight]
        for ii, res_modes in enumerate(fitter):
            DXY = np.zeros_like(freqs)
            DXYerr = np.zeros_like(freqs)
            DZ = np.zeros_like(freqs)
            DZerr = np.zeros_like(freqs)
            for kk, fit_ in enumerate(res_modes):
                k_0 = k0(freqs[kk])
                DZ[kk] = D0 - 100 * fit_.params["DZ"].value / k_0
                DZerr[kk] = 100 * fit_.params["DZ"].stderr / k_0
            ax.errorbar(
                freqs,
                DZ,
                DZerr,
                color=cmap(ii),
                marker=mm[jj],
                label=f"{legend[jj]} - {modes[ii]}",
                **kwargs,
            )
            ax.set_xlabel(r"Frequency (GHz)", fontsize=8)
            ax.set_ylabel(r"$\Delta_{Z} / k_0$ (Physical distance (cm))r", fontsize=8)
            ax.legend(fontsize=8)
            # )

    return ax
