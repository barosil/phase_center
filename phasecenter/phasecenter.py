import numpy as np
import pandas as pd
from scipy.constants import c
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

DATASET = [
    "beampattern_horn01_Polarização_Horizontal_Copolar.csv",
    "beampattern_horn01_Polarização_Vertical_Copolar.csv",
    # "beampattern_horn01_Polarização_Horizontal_Cruzada.csv",
    # "beampattern_horn01_Polarização_Vertical_Cruzada.csv",
]


def get_color(name):
    pol = name[0]
    if pol == "Horizontal_Copolar":
        return "blue"
    else:
        return "red"


def get_linewidth(name):
    weight = name[1]
    if weight == "Amplitude":
        return 1.0
    else:
        return 0.5


def get_linestyle(name):
    smooth = name[2]
    if smooth:
        return "--"
    else:
        return "-"


class PhaseCenter:
    def __init__(
        self,
        bootstrap=True,
        smooth=True,
        dataset=DATASET,
        theta_cut=20,
        taper=-10,
        D0=2.35,
        path="../data/raw/",
    ):
        self.path = path
        self._dataset = dataset
        self.theta_cut = theta_cut
        self.taper = taper
        self.bootstrap = bootstrap
        self.smooth = smooth
        self.guess = None
        self.data = None
        self.params = pd.DataFrame()
        self.best_fit = None
        self._D0 = D0

    @property
    def D0(self):
        return self._D0

    @D0.setter
    def D0(self, D0):
        self._D0 = D0

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def _normalize_AMP(self, data):
        data.AMPLITUDE = data.AMPLITUDE - data.AMPLITUDE.max()
        Theta_0 = data.ANGLE[data.AMPLITUDE.idxmax()]
        data.ANGLE = data.ANGLE - Theta_0
        return data

    def _normalize_PHASE(self, data):
        PHI_0 = data[data.ANGLE == 0]["PHASE"].values[0]
        data["PHASE"] = np.unwrap(data.PHASE - PHI_0)
        return data

    def _get_taper_angle(self, data):
        data_interp = sp.interpolate.interp1d(data["ANGLE"], data["AMPLITUDE"])
        taper_angle = sp.optimize.minimize(
            lambda angle: np.abs(data_interp(angle) - self.taper),
            np.radians(10),
            method="nelder-mead",
        ).x[0]
        return taper_angle * np.ones(data["ANGLE"].shape)

    def _get_angle_phase(self, data):
        ph = sp.signal.savgol_filter(data.PHASE.values, 5, 2)
        peaks, _ = sp.signal.find_peaks(-ph, width=10)
        theta_peak = data.ANGLE.iloc[peaks].abs().min()
        return theta_peak * np.ones(data["ANGLE"].shape)

    def _load_dataset(self, file):
        data = (
            pd.read_csv(file)
            .query("ANGLE > -@self.theta_cut & ANGLE < @self.theta_cut")
            .reset_index(drop=True)
        )
        dataset = "_".join(file.split("/")[-1].split("_")[-2:]).split(".")[0]
        data[["ANGLE", "PHASE"]] = np.radians(data[["ANGLE", "PHASE"]])
        data_ = (
            data.groupby("FREQ")[["FREQ", "ANGLE", "PHASE", "AMPLITUDE"]]
            .apply(self._normalize_AMP)
            .reset_index(drop=True)
            .groupby("FREQ")[["FREQ", "ANGLE", "PHASE", "AMPLITUDE"]]
            .apply(self._normalize_PHASE)
            .reset_index(drop=True)
            .groupby("FREQ")[["ANGLE", "PHASE", "AMPLITUDE"]]
            .apply(
                lambda data: data.assign(FWHM=lambda data: self._get_taper_angle(data))
            )
            .reset_index()
            .groupby("FREQ")[["FREQ", "ANGLE", "PHASE", "AMPLITUDE", "FWHM"]]
            .apply(
                lambda data: data.assign(
                    THETA_CUT=lambda data: self._get_angle_phase(data)
                )
            )
            .reset_index(drop=True)
            .groupby("FREQ")[
                ["FREQ", "ANGLE", "PHASE", "AMPLITUDE", "FWHM", "THETA_CUT"]
            ]
            .apply(
                lambda data: data.assign(
                    THETA_MAX=lambda data: np.min([data.FWHM, data.THETA_CUT])
                )
            )
            .query("ANGLE >= -THETA_MAX & ANGLE <= THETA_MAX")
            .reset_index(drop=True)
        )
        data_["DATASET"] = dataset
        if self.smooth:
            data_.groupby(["DATASET", "FREQ"])[data_.columns].apply(
                lambda data: data.assign(PHASE_SM=self._smooth_func(data.PHASE))
            ).reset_index(drop=True)
        return data_

    def _load_data(self):
        filenames = [self.path + file for file in self.dataset]
        data = pd.concat([self._load_dataset(filename) for filename in filenames])
        self.data = data
        return self

    def _smooth_func(self, phase):
        return sp.signal.savgol_filter(phase.values, 20, 2)

    def _smooth_phases(self, data):
        phases = []
        for gr, group in data.groupby(["DATASET", "FREQ"]):
            phases.append(self._smooth_func(group.PHASE))
        phases = np.concatenate(phases)
        return phases

    def fit_phase(self, model=None, bootstrap=None, smooth=None, sigma_func=None):
        if bootstrap is None:
            bootstrap = self.bootstrap
        if smooth is None:
            smooth = self.smooth
        if sigma_func is None:
            sigma_func = PhaseCenter._sigma_Amp
        if model is None:
            model = PhaseCenter.model
        cols_orig = ["DATASET", "FREQ", 0, 1, 2, 3, 4, 5]
        par_names = [
            "DATASET",
            "FREQ",
            "DZ",
            "PHI_0",
            "DXY",
            "DZ_err",
            "PHI_0_err",
            "DXY_err",
        ]
        if smooth:
            phase_str = "PHASE_SM"
            if phase_str not in self.data.columns:
                self.data = (
                    self.data.groupby(["DATASET", "FREQ"])[self.data.columns]
                    .apply(
                        lambda data: data.assign(PHASE_SM=self._smooth_func(data.PHASE))
                    )
                    .reset_index(drop=True)
                )
        else:
            phase_str = "PHASE"

        while True:
            fit = (
                self.data.groupby(["DATASET", "FREQ"])[
                    ["ANGLE", phase_str, "AMPLITUDE"]
                ]
                .apply(
                    lambda data: pd.Series(
                        self._fit_phase_center(
                            data,
                            model,
                            sigma_func,
                            phase_str=phase_str,
                            bootstrap=bootstrap,
                            group=data.name,
                        )
                    )
                )
                .reset_index()
                .rename(columns=dict(zip(cols_orig, par_names)))
            )
            fit["SMOOTH"] = smooth
            weight = "Amplitude" if sigma_func == PhaseCenter._sigma_Amp else "Uniform"
            fit["WEIGHT"] = weight
            if self.params.empty:
                fit["SAMPLE"] = 1
            else:
                groups = self.params.groupby(["WEIGHT", "SMOOTH"]).groups.keys()
                if (weight, smooth) in groups:
                    fit["SAMPLE"] = (
                        self.params[
                            (self.params.WEIGHT == weight)
                            & (self.params.SMOOTH == smooth)
                        ].SAMPLE.max()
                        + 1
                    )
                else:
                    fit["SAMPLE"] = 1
            if bootstrap:
                yield fit
            else:
                return fit

    def set_guess(self):
        self.guess = next(self.fit_phase())
        return self

    def _dz_phys(self, data, D0=2.35):
        data["DZ_phys"] = 100 * (D0 - data["DZ"] / PhaseCenter._k0(data["FREQ"]))
        return data

    def _wavelength_cm(self, data):
        data["WAVELENGTH"] = 100 * c / (data["FREQ"] * 1e9)
        return data

    def _wrap_angle(theta: np.ndarray) -> np.ndarray:
        # result = (theta + np.pi) % (2 * np.pi) - np.pi
        result = np.arctan2(np.sin(theta), np.cos(theta))
        return result

    def _k0(freq: float) -> float:
        k0 = 2 * np.pi * freq * 1e9 / c
        return k0

    def _sigma_uniform(data) -> np.ndarray:
        return np.ones(data.size)

    def _sigma_Amp(data) -> np.ndarray:
        return 1 / (10 ** (data / 10))

    def model(theta, DZ, PHI0, DXY: float) -> np.ndarray:
        Phi = DXY * np.sin(theta) + DZ * np.cos(theta) + PHI0
        return Phi

    def _fit_phase_center(
        self, data, model, sigma_func, phase_str="PHASE", bootstrap=False, group=None
    ):
        if bootstrap:
            idx = np.random.choice(data.index, size=len(data), replace=True)
            data = data.loc[idx]
        if self.guess is None:
            guess = [0, 0, 0]
        else:
            guess = self.guess[
                (self.guess.DATASET == group[0]) & (self.guess.FREQ == group[1])
            ][["DZ", "PHI_0", "DXY"]].values[0]
        sigma = sigma_func(data.AMPLITUDE)
        popt, pcov = sp.optimize.curve_fit(
            model, data.ANGLE, data[phase_str], p0=guess, sigma=sigma, maxfev=10000
        )
        param = popt
        perr = np.sqrt(np.diag(pcov))
        return [*param, *perr]

    def run_bootstrap(self, sigma_func=None, smooth=True, n=100):
        if sigma_func is None:
            sigma_func = PhaseCenter._sigma_Amp
        for ii in range(n):
            if self.params.empty:
                self.params = next(self.fit_phase(sigma_func=sigma_func, smooth=smooth))
            else:
                self.params = pd.concat(
                    [
                        self.params,
                        next(self.fit_phase(sigma_func=sigma_func, smooth=smooth)),
                    ]
                )
        return self

    def _best_fit(self, data, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        n_params = data.shape[1]
        if n_params == 1:
            _data = (data.values,)
            res = sp.stats.bootstrap(_data, np.mean, random_state=rng)
            values = np.asarray(
                [
                    data.mean()[0],
                    res.confidence_interval[0][0],
                    res.confidence_interval[1][0],
                ]
            )
        else:
            values = np.zeros((n_params, 3))
            for ii in range(n_params):
                _data = (data.iloc[:, ii].values,)
                res = sp.stats.bootstrap(_data, np.mean, random_state=rng)
                values[ii, :] = np.asarray(
                    [
                        data.iloc[:, ii].mean(),
                        res.confidence_interval[0],
                        res.confidence_interval[1],
                    ]
                )
            values = np.ravel(values)
        return values

    def run_best_fit(self):
        cols_orig = ["DATASET", "WEIGHT", "SMOOTH", "FREQ", 0, 1, 2, 3, 4, 5, 6, 7, 8]
        bv_names = [
            "DATASET",
            "WEIGHT",
            "SMOOTH",
            "FREQ",
            "DZ",
            "DZ_err_low",
            "DZ_err_high",
            "PHI_0",
            "PHI_0_err_low",
            "PHI_0_err_high",
            "DXY",
            "DXY_err_low",
            "DXY_err_high",
        ]
        best_fit = (
            self.params.groupby(["DATASET", "WEIGHT", "SMOOTH", "FREQ"])[
                ["DZ", "PHI_0", "DXY"]
            ]
            .apply(lambda data: pd.Series(self._best_fit(data)))
            .reset_index()
            .rename(columns=dict(zip(cols_orig, bv_names)))
        )
        self.best_fit = best_fit
        return self

    def _predict(self, params, group):
        angles = self.data.query("DATASET == @group[0] & FREQ == @group[-1]")[
            "ANGLE"
        ].values
        guess = params.values[0]
        result = PhaseCenter.model(angles, *guess)

        return result

    def predict(self):
        res = self.best_fit.copy()
        # res[["WEIGHT", "SMOOTH", "PREDICTED"]] = (
        #     res.groupby(["DATASET", "FREQ"])[["ANGLE", "PHASE"]]
        #     .apply(lambda data: self._cal_model(data, group=data.name))
        #     .reset_index(drop=True)
        # )
        params = self.best_fit.groupby(["DATASET", "WEIGHT", "SMOOTH", "FREQ"])[
            ["DZ", "PHI_0", "DXY"]
        ]
        results = []
        for group, param in params:
            angles = self.data.query("DATASET == @group[0] & FREQ == @group[-1]")[
                "ANGLE"
            ].values
            phases = self.data.query("DATASET == @group[0] & FREQ == @group[-1]")[
                "PHASE"
            ].values
            guess = param.values[0]
            predicted = PhaseCenter.model(angles, *guess)
            result = pd.DataFrame(
                {
                    "DATASET": group[0],
                    "WEIGHT": group[1],
                    "SMOOTH": group[2],
                    "FREQ": group[3],
                    "ANGLE": angles,
                    "PHASE": phases,
                    "PREDICTED": predicted,
                }
            )
            results.append(result)
        self.predicted = pd.concat(results)
        return self

    def score_R2(data):
        return 1 - np.sum((data.PHASE - data.PREDICTED) ** 2) / np.sum(
            (data.PHASE - data.PHASE.mean()) ** 2
        )

    def score_Chi2(data):
        return np.sum((data.PHASE - data.PREDICTED) ** 2 / (data.shape[0] - 3))

    def score_p(data):
        return sp.stats.chisquare(data.PHASE, data.PREDICTED)[1]

    def test_cramer(data):
        try:
            res = sp.stats.cramervonmises_2samp(data.PHASE, data.PREDICTED)
        except ValueError:
            return np.nan
        return res.pvalue

    def test_KS(data):
        res = sp.stats.ks_2samp(data.PHASE, data.PREDICTED)[1]
        return res

    def test_KS_res(data):
        res = data.PHASE - data.PREDICTED
        test = sp.stats.kstest(res, sp.stats.norm.cdf)[1]
        return test

    def score(self):
        res = (
            self.predicted.groupby(["DATASET", "WEIGHT", "SMOOTH", "FREQ"])
            .apply(
                lambda data: pd.Series(
                    {
                        "R2": PhaseCenter.score_R2(data),
                        "Chi2": PhaseCenter.score_Chi2(data),
                        "cramer": PhaseCenter.test_cramer(data),
                        "KS": PhaseCenter.test_KS(data),
                        "KS_res": PhaseCenter.test_KS_res(data),
                    }
                )
            )
            .reset_index()
        )
        self.best_fit = pd.merge(
            self.best_fit, res, on=["DATASET", "WEIGHT", "SMOOTH", "FREQ"], how="inner"
        )
        return self

    def report(self):
        self.best_fit["Wavelength_cm"] = 100 * c / (self.best_fit["FREQ"] * 1e9)
        self.best_fit["DZ"] = (
            self.D0 / self.best_fit.Wavelength_cm - self.best_fit["DZ"] / 2 / np.pi
        )
        self.best_fit["DZ_err_low"] = np.abs(
            self.best_fit["DZ_err_low"] / 2 / np.pi - self.best_fit["DZ"]
        )
        self.best_fit["DZ_err_high"] = np.abs(
            self.best_fit["DZ_err_high"] / 2 / np.pi - self.best_fit["DZ"]
        )
        self.best_fit["DXY"] = np.abs(self.best_fit["DXY"] / 2 / np.pi)
        self.best_fit["DXY_err_low"] = np.abs(
            self.best_fit["DXY_err_low"] / 2 / np.pi - self.best_fit["DXY"]
        )
        self.best_fit["DXY_err_high"] = np.abs(
            self.best_fit["DXY_err_high"] / 2 / np.pi - self.best_fit["DXY"]
        )
        self.best_fit["DZ_phys"] = self.best_fit["DZ"] * self.best_fit.Wavelength_cm
        self.best_fit["DZ_err_low_phys"] = np.abs(
            self.best_fit["DZ_err_low"] * self.best_fit.Wavelength_cm
            - self.best_fit["DZ_phys"]
        )
        self.best_fit["DZ_err_high_phys"] = np.abs(
            self.best_fit["DZ_err_high"] * self.best_fit.Wavelength_cm
            - self.best_fit["DZ_phys"]
        )
        self.best_fit["DXY_phys"] = self.best_fit["DXY"] * self.best_fit.Wavelength_cm
        self.best_fit["DXY_err_low_phys"] = np.abs(
            self.best_fit["DXY_err_low"] * self.best_fit.Wavelength_cm
            - self.best_fit["DXY_phys"]
        )
        self.best_fit["DXY_err_high_phys"] = np.abs(
            self.best_fit["DXY_err_high"] * self.best_fit.Wavelength_cm
            - self.best_fit["DXY_phys"]
        )
        self.best_fit["PHI_0"] = np.degrees(self.best_fit["PHI_0"])
        self.best_fit["PHI_0_err_low"] = np.degrees(self.best_fit["PHI_0_err_low"])
        self.best_fit["PHI_0_err_high"] = np.degrees(self.best_fit["PHI_0_err_high"])
        self.best_fit["Wavelength_cm"] = np.round(self.best_fit["Wavelength_cm"], 2)
        return self

    def save(self, path="../data/processed"):
        timestamp = pd.Timestamp.now().strftime("%Y_%_m%d_%H_%M_%S")
        self.best_fit.to_csv(f"{path}/best_fit_{timestamp}.csv", index=False)
        self.predicted.to_csv(f"{path}/predicted_{timestamp}.csv", index=False)
        self.data.to_csv(f"{path}/data_{timestamp}.csv", index=False)
        return self

    def plot_phase_center(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 4))
        for name, data in self.best_fit.groupby(["DATASET", "WEIGHT", "SMOOTH"]):
            yerr = data[["DZ_err_low", "DZ_err_high"]].T.values
            yerr_phys = data[["DZ_err_low_phys", "DZ_err_high_phys"]].T.values
            ax[0, 0].errorbar(
                1000 * data.FREQ,
                data.DZ,
                yerr=yerr,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[0, 1].errorbar(
                1000 * data.FREQ,
                data.DZ_phys,
                yerr=yerr_phys,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[0, 0].set_xlabel("Frequency (MHz)")
            ax[0, 0].set_ylabel(r"Phase Center $\Delta_z$ ($\lambda$)")
            ax[0, 1].set_xlabel("Frequency (MHz)")
            ax[0, 1].set_ylabel(r"Phase Center $\Delta_z$ (cm)")
            yerr = data[["DXY_err_low", "DXY_err_high"]].T.values
            yerr_phys = data[["DXY_err_low_phys", "DXY_err_high_phys"]].T.values
            ax[1, 0].errorbar(
                1000 * data.FREQ,
                data.DXY,
                yerr=yerr,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[1, 1].errorbar(
                1000 * data.FREQ,
                data.DXY_phys,
                yerr=yerr_phys,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[1, 0].set_xlabel("Frequency (MHz)")
            ax[1, 0].set_ylabel(r"Phase Center $\Delta_{XY}$ ($\lambda$)")
            ax[1, 1].set_xlabel("Frequency (MHz)")
            ax[1, 1].set_ylabel(r"Phase Center $\Delta_{XY}$ (cm)")
            ax[1, 0].legend(loc="lower right", bbox_to_anchor=(1.8, -0.5), ncol=2)

        return ax

    def plot_statistics(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 2 * 3))

        for name, data in self.best_fit.groupby(["DATASET", "WEIGHT", "SMOOTH"]):
            ax[0, 0].plot(
                1000 * data.FREQ,
                data.R2,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[0, 1].plot(
                1000 * data.FREQ,
                data.cramer,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[1, 0].plot(
                1000 * data.FREQ,
                data.KS,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[1, 1].plot(
                1000 * data.FREQ,
                data.KS_res,
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )
            ax[0, 0].set_xlabel("Frequency (MHz)")
            ax[0, 0].set_ylabel(r"Coefficient of determination $R^2$")
            ax[0, 1].set_xlabel("Frequency (MHz)")
            ax[0, 1].set_ylabel("Cramer-von Mises p-value")
            ax[1, 0].set_xlabel("Frequency (MHz)")
            ax[1, 0].set_ylabel("Kolmogorov-Smirnov p-value")
            ax[1, 1].set_xlabel("Frequency (MHz)")
            ax[1, 1].set_ylabel("Kolmogorov-Smirnov p-value (residuals)")
            ax[2, 0].set_xlabel("Frequency (MHz)")
            ax[2, 0].set_ylabel(r"$\chi^2$")
            ax[2, 0].plot(
                1000 * data.FREQ,
                data.Chi2,
                color=get_color(name),
                linewidth=get_linewidth(name),
                linestyle=get_linestyle(name),
                label=f"Dataset: {name[0]}, Weights: {name[1]}, Smooth: {name[2]}",
            )

            ax[2, 0].legend(loc="lower right", bbox_to_anchor=(1.8, -0.4), ncol=2)
        return ax

    def plot_phases(self, freqs=None, ncols=3, ax=None):
        if freqs is None:
            freqs = self.best_fit.FREQ.unique()
        ncols = 4
        nrows = int(np.ceil(2 * len(freqs) / ncols))
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(8, 2 * nrows),
                gridspec_kw={"hspace": 0, "wspace": 0},
            )
        jj = 0
        kk = 0
        for pol, dataset in self.predicted.groupby(["DATASET"]):
            for name_, dataset_ in dataset.groupby(["WEIGHT", "SMOOTH"]):
                for ii, freq in enumerate(freqs):
                    kk = ii + jj
                    name = (pol[0], name_[0], name_[1])
                    data = dataset_[dataset_.FREQ == freq].sort_values("ANGLE").copy()
                    ax[kk // ncols, kk % ncols].plot(
                        np.degrees(data.ANGLE),
                        np.degrees(data.PHASE),
                        color="black",
                        linewidth=1.5,
                        linestyle=":",
                        alpha=0.5,
                        label=f"Measured",
                    )
                    phase_smooth = self.data.query(
                        f"FREQ == {freq} and DATASET == '{pol[0]}' and ANGLE <= {data.ANGLE.max()} and ANGLE >= {-data.ANGLE.max()}"
                    )[["ANGLE", "PHASE_SM"]]
                    ax[kk // ncols, kk % ncols].plot(
                        np.degrees(phase_smooth.ANGLE),
                        np.degrees(phase_smooth.PHASE_SM),
                        color="violet",
                        linewidth=1.5,
                        alpha=0.5,
                        label=f"Smoothed - Savitsky-Golay (Loess)",
                    )
                    ax[kk // ncols, kk % ncols].plot(
                        np.degrees(data.ANGLE),
                        np.degrees(data.PREDICTED),
                        color=get_color(name),
                        linewidth=get_linewidth(name),
                        linestyle=get_linestyle(name),
                        label=f"Predicted - Dataset: {pol[0]}, Weights: {name[1]}, Smooth: {name[2]}",
                        alpha=0.3,
                    )
                    ax[kk // ncols, kk % ncols].set_title(
                        f"Frequency: {freq * 1000:.0f} MHz - {pol[0]}",
                    )
                    ax[kk // ncols, kk % ncols].grid(
                        color="gray", linestyle="--", linewidth=0.2
                    )

            jj = kk + 1

        ax[0, 0].set_xlabel(
            r"$\theta$ (degrees)",
        )
        ax[nrows - 1, 0].set_ylabel(
            r"phase $\phi$ (degrees)",
        )

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        # grab unique labels
        unique_labels = set(labels)

        # assign labels and legends in dict
        legend_dict = dict(zip(labels, lines))

        # query dict based on unique labels
        unique_lines = [legend_dict[x] for x in unique_labels]

        ax[nrows - 1, 0].legend(
            unique_lines,
            unique_labels,
            loc="lower right",
            bbox_to_anchor=(3.5, -0.6),
            ncol=2,
        )
        return ax

    def plot_residuals(self, freqs=None, ncols=3, ax=None):
        if freqs is None:
            freqs = self.best_fit.FREQ.unique()
        ncols = 4
        nrows = int(np.ceil(len(freqs) / ncols))
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(8, 2 * nrows),
                gridspec_kw={"hspace": 0, "wspace": 0},
            )
        markers = [
            "o",
            "s",
            "D",
            "x",
            ".",
            "^",
            ">",
            "<",
            "v",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "X",
            "D",
            "d",
        ]
        alpha = 0.5
        dataset = self.predicted
        for name_, dataset_ in dataset.groupby(["DATASET", "WEIGHT", "SMOOTH"]):
            for ii, freq in enumerate(freqs):
                data = dataset_[dataset_.FREQ == freq].sort_values("ANGLE").copy()
                ll = list(
                    dataset.groupby(["DATASET", "WEIGHT", "SMOOTH"]).groups.keys()
                ).index(name_)
                p1 = sm.ProbPlot(data.PHASE)
                p2 = sm.ProbPlot(data.PREDICTED)
                p1.qqplot(
                    other=p2,
                    line="45",
                    xlabel="",
                    ylabel="",
                    **{
                        "markersize": 2,
                        "marker": markers[ll],
                        "alpha": alpha,
                        "label": f"Predicted - Dataset: {name_[0]}, Weights: {name_[1]}, Smooth: {name_[2]}",
                    },
                    ax=ax[ii // ncols, ii % ncols],
                )
                ax[ii // ncols, ii % ncols].grid(
                    color="gray", linestyle="--", linewidth=0.2
                )
                ax[ii // ncols, ii % ncols].set_title(
                    f"Frequency: {freq * 1000:.0f} MHz",
                )
        ax[0, 0].set_xlabel(
            xlabel="Measured Phase",
        )
        ax[nrows - 1, 0].set_ylabel(
            ylabel="Predicted Phase",
        )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        # grab unique labels
        unique_labels = set(labels)

        # assign labels and legends in dict
        legend_dict = dict(zip(labels, lines))

        # query dict based on unique labels
        unique_lines = [legend_dict[x] for x in unique_labels]

        ax[nrows - 1, 0].legend(
            unique_lines,
            unique_labels,
            loc="lower right",
            bbox_to_anchor=(3.5, -0.6),
            ncol=2,
        )
        return ax

    def plot_qq_residuals(self, freqs=None, ncols=4, ax=None):
        markers = ["1", "o", "+", "D"]
        freqs = self.best_fit.FREQ.unique()[::4]
        if freqs is None:
            freqs = self.best_fit.FREQ.unique()
        ncols = 4
        nrows = int(np.ceil(2 * len(freqs) / ncols))

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            figsize=(8, 2 * nrows),
        )
        kk = 0
        jj = 0
        for pol, dataset in self.predicted.groupby(["DATASET"]):
            for name_, dataset_ in dataset.groupby(["DATASET", "WEIGHT", "SMOOTH"]):
                for ii, freq in enumerate(freqs):
                    kk = ii + jj
                    ll = list(
                        dataset.groupby(["DATASET", "WEIGHT", "SMOOTH"]).groups.keys()
                    ).index(name_)
                    name = (pol[0], name_[0], name_[1])
                    data = dataset_[dataset_.FREQ == freq]
                    res = data.apply(
                        lambda data: data.PHASE - data.PREDICTED, axis=1
                    ).values

                    sm.qqplot(
                        res,
                        line="s",
                        marker=markers[ll],
                        markersize=2,
                        ax=ax[kk // ncols, kk % ncols],
                        label=f"{name_[1]} {name_[2]}",
                    )

                    ax[kk // ncols, kk % ncols].grid(
                        color="gray", linestyle="--", linewidth=0.2
                    )
                    ax[kk // ncols, kk % ncols].set_title(
                        f"{freq * 1000:.0f} MHz - {name[0]}",
                    )
                    ax[kk // ncols, kk % ncols].set_xlabel(
                        xlabel="",
                    )
                    ax[kk // ncols, kk % ncols].set_ylabel(
                        ylabel="",
                    )
                    [
                        item.set_fontsize(5)
                        for item in ax[kk // ncols, kk % ncols].get_xticklabels()
                    ]
                    [
                        item.set_fontsize(5)
                        for item in ax[kk // ncols, kk % ncols].get_yticklabels()
                    ]
            jj = kk + 1
        ax[nrows - 1, 0].set_xlabel(
            xlabel="Theoretical Quantiles",
        )
        ax[nrows - 1, 0].set_ylabel(
            ylabel="Residues",
        )

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

        # grab unique labels
        unique_labels = set(labels)

        # assign labels and legends in dict
        legend_dict = dict(zip(labels, lines))

        # query dict based on unique labels
        unique_lines = [legend_dict[x] for x in unique_labels]

        ax[nrows - 1, 0].legend(
            unique_lines,
            unique_labels,
            loc="lower right",
            bbox_to_anchor=(4, -0.5),
            ncol=4,
        )
        return ax

    def get_recent_file(path, mask):
        files = list(Path(path).glob(mask))
        file = max(files, key=lambda file: file.stat().st_ctime)
        return file

    def load(self, path, files=None):
        if files is None:
            masks = ["best_fit*.csv", "predicted*.csv", "data*.csv"]
            files = [PhaseCenter.get_recent_file(path, mask) for mask in masks]
        self.best_fit = pd.read_csv(files[0])
        self.predicted = pd.read_csv(files[1])
        self.data = pd.read_csv(files[2])
        return self
