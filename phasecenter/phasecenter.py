import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c
import scipy as sp

def dz_phis(dzs, D0=2.35):
    freqs = get_freqs()
    return 100 * (D0 - dzs /k0(freqs))

def get_freqs(mode="V", path="../data/raw/"):
    dataset_H = path + "beampattern_horn01_Polarização_Horizontal_Copolar.csv"
    dataset_V = path + "beampattern_horn01_Polarização_Vertical_Copolar.csv"
    if mode == "V":
        data = pd.read_csv(dataset_V)
    else:
        data = pd.read_csv(dataset_H)
    return data.FREQ.unique()

def get_taper_angle(taper, freq, degrees=True):
    data = get_data(freq)
    data_interp = sp.interpolate.interp1d(data['ANGLE'], data['AMPLITUDE'])
    taper_angle = sp.optimize.minimize(lambda angle: np.abs(data_interp(angle) - taper), np.radians(10), method="nelder-mead").x[0]
    if degrees:
        taper_angle = np.degrees(taper_angle)
    return taper_angle

def get_theta_range(taper, degrees=True):
    freqs = get_freqs()
    pcs = []
    for freq in freqs:
        FWHM = get_taper_angle(taper, freq, degrees=False)
        data = get_data(freq).query("ANGLE < @FWHM and ANGLE > -@FWHM").reset_index(drop=True)
        PHI_0 = data.PHASE.iloc[data.AMPLITUDE.idxmax()]
        theta_0 = get_max_amp(data)
        data["ANGLE"] = data.ANGLE - theta_0
        data["PHASE"] = data.PHASE - PHI_0
        pc_range = min(FWHM, data[data.PHASE > 1].ANGLE.abs().min())
        if degrees:
            pc_range = np.degrees(pc_range)
        pcs.append(pc_range)
    
    return pcs

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


def normalize_PHASE(data):
    data = data.copy()
    theta_0 = get_max_amp(data)
    PHI_0 = data.PHASE.iloc[data.AMPLITUDE.idxmax()]
    data["ANGLE"] = data.ANGLE - theta_0
    data["PHASE"] = data.PHASE - PHI_0
    data["delta_theta"] = theta_0
    
    return data


def get_data(
    freq=1.0, mode="V", theta_max_deg=None, smooth=True, path="../data/raw/"):
    dataset_H = path + "beampattern_horn01_Polarização_Horizontal_Copolar.csv"
    dataset_V = path + "beampattern_horn01_Polarização_Vertical_Copolar.csv"
    dataset_XH = path + "beampattern_horn01_Polarização_Horizontal_Cruzada.csv"
    dataset_XV = path + "beampattern_horn01_Polarização_Vertical_Cruzada.csv"
    if mode == "V":
        data = pd.read_csv(dataset_V)
    elif mode == "H":
        data = pd.read_csv(dataset_H)
    elif mode == "XH":
        data = pd.read_csv(dataset_XH)
    elif mode == "XV":
        data = pd.read_csv(dataset_XV)
    else:
        print("Invalid mode")
    data = data.query("FREQ==@freq").copy().reset_index(drop=True)
    data.ANGLE = np.radians(data.ANGLE)
    data.PHASE = np.unwrap(np.radians(data.PHASE))
    if theta_max_deg is not None:
        data = cut_theta(data, theta_max_deg)
    # Normalize amplitudes such that the maximum amplitude is 0 dB.
    data = normalize_AMP(data)
    # Normalize phases such that the phase at the maximum amplitude is 0.
    data = normalize_PHASE(data)
    if smooth:
        data.PHASE = sp.signal.savgol_filter(data.PHASE, 10, 2)
    return data


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    #result = (theta + np.pi) % (2 * np.pi) - np.pi
    result = np.arctan2(np.sin(theta), np.cos(theta))
    return result


def k0(freq: float) -> float:
    k0 = 2 * np.pi * freq * 1e9 / c
    return k0


def weight_uniform(data) -> np.ndarray:
    return np.ones(data.size)


def weight_Amp(data) -> np.ndarray:
    return 10 ** (data / 10)


def model(theta, DZ, PHI0, DXY: float) -> np.ndarray:
    Phi = DXY * np.sin(theta) + DZ * np.cos(theta) + PHI0
    return Phi

def model_even(theta, DZ: float, PHI0) -> np.ndarray:
    Phi = DZ * np.cos(theta) + PHI0
    return Phi

def model_odd(theta, DXY: float) -> np.ndarray:
    Phi = DXY * np.sin(theta)
    return Phi


def fit_phase_center(DZ=0, theta_max=None, taper=-20, mode="V", smooth=True):
    _freqs = get_freqs()
    if taper is not None:
        theta_range = get_theta_range(taper)
    elif theta_max is not None:
        theta_range = [theta_max] * len(_freqs)
    else:
        theta_range = [90] * len(_freqs)
    params = []
    params_err = []
    for ii, freq in enumerate(_freqs):
        try:
            data = get_data(freq, theta_max_deg=theta_range[ii], mode=mode, smooth=smooth)
            guess = [DZ, 0, 0]
            popt, pcov = sp.optimize.curve_fit(model, data.ANGLE, data.PHASE, p0=guess, maxfev=10000)
            params.append(popt)
            params_err.append(np.sqrt(np.diag(pcov)))
        except ValueError:
            pass
    result = np.hstack([np.asarray(params), np.asarray(params_err)])
    return result


def get_phase_center(DZ=0, theta_max=None, taper=-20, smooth=True):
    if not theta_max:
        theta_max = [20]
    result = np.zeros((len(theta_max), 3), dtype=object)
    for ii, theta in enumerate(theta_max):
        result_H = fit_phase_center(DZ=DZ, theta_max=theta, taper=taper, mode="H", smooth=smooth)
        result_V = fit_phase_center(DZ=DZ, theta_max=theta, taper=taper, mode="V", smooth=smooth)
        result[ii, :] = [theta, result_H, result_V]
    return result

def plot_phase_center(params, ax=None):
    freqs = get_freqs()
    n_sets = params.shape[0]
    points_H = [ [params[jj, 1][:, 0], params[jj, 1][:, 3]] for jj in range(n_sets) ]
    sets = params[:, 0]
    points_V = [ [params[jj, 2][:, 0], params[jj, 2][:, 3]] for jj in range(n_sets) ]
    if ax is None:
        fig, ax = plt.subplots()
    for dz_H, dz_V in zip(points_H, points_V):
        for ii, Set in enumerate(sets):
            ax.errorbar(freqs, dz_phis(dz_H[0]), 100 * dz_H[1], label=f"Horizontal", linewidth=1 / (ii + 1), color="blue")
            ax.errorbar(freqs, dz_phis(dz_V[0]), 100 * dz_V[1], label=f"Vertical", linewidth=1 / (ii + 1), color="red")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Phase Center (cm)")
    ax.legend()
    return ax

def plot_phases(freq, params, D0=2.35, label="Horizontal Copolar", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    freqs = get_freqs()
    idx = np.where(freqs == freq)[0][0]
    theta_range = get_theta_range(-20)[idx]
    data = get_data(freq, theta_max_deg=theta_range)
    angles = np.degrees(data.ANGLE)
    phases = np.degrees(data.PHASE)
    model_ = np.degrees(model(data.ANGLE, *params))
    
    ax.plot(angles, phases, label=f"measured {label}", color="blue", linewidth=1.5)
    DZ = f"{100 * (D0 - params[0] / k0(freq)):.0f} cm"
    ax.plot(angles, model_, label=r"fitted $\Delta_z= $" + DZ, linestyle="--", color="red")
    ax.set_xlabel("Angle [deg]", fontsize=6)
    ax.set_ylabel("Phase [deg]", fontsize=6)
    ax.set_title(f"{1000 * freq:.0f} MHz - " + label + r"  - $\theta_{MAX}= $" + f"{theta_range:.0f}", fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    return ax

