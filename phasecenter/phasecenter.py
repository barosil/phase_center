from __future__ import annotations

import functools
import itertools
import operator
from pathlib import Path
from typing import Any, Callable, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from scipy.constants import c

DATASET = [
    "beampattern_horn01_Polarização_Horizontal_Copolar.csv",
    "beampattern_horn01_Polarização_Vertical_Copolar.csv",
#    "beampattern_horn01_Polarização_Horizontal_Cruzada.csv",
#    "beampattern_horn01_Polarização_Vertical_Cruzada.csv",
]


class PhaseCenter:
    def __init__(
        self,
        dataset: list[str] = DATASET,
        theta_cut: int = 20,
        taper: int = -10,
        D0: float = 2.35,
        path: str = "../data/raw/",
        sigma_theta: float | None = None,
        sigma_phase: float | None = None,
    ) -> None:
        self.dataset = dataset
        self._D0 = D0
        self.path = path
        self._dataset = dataset
        self.sigma_theta = sigma_theta
        self.sigma_phase = sigma_phase
        self.theta_cut = theta_cut
        self.taper = taper
        self.data = self.load_data()
        self.params = pd.DataFrame()
        self.best_fit = None
        self.GROUPS = [
            "DATASET",
            "WEIGHT",
            "SMOOTH",
            "BOOTSTRAP",
            "METHOD",
            "FREQ",
        ]

    @property
    def D0(self) -> float:
        return self._D0

    @D0.setter
    def D0(self, D0: float) -> None:
        self._D0 = D0

    @property
    def dataset(self) -> list[str]:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: list[str]) -> None:
        self._dataset = dataset

    @staticmethod
    def _wavelength_cm(freq: float) -> float:
        return 100 * c / (freq * 1e9)

    @staticmethod
    def _DZ_lam(DZ: float, D0: float, freq: float) -> float:
        return 100 * D0 / PhaseCenter._wavelength_cm(freq) - DZ / 2 / np.pi

    @staticmethod
    def _DZ_err_lam(DZ: float, DZ_err: float, freq: float, D0: float) -> float:
        return (DZ_err / DZ) * PhaseCenter._DZ_lam(DZ, D0, freq)

    @staticmethod
    def _DZ_phys(DZ: float, D0: float, freq: float) -> float:
        return PhaseCenter._DZ_lam(DZ, D0, freq) * PhaseCenter._wavelength_cm(
            freq,
        )

    def _normalize_AMP(self, data: pd.DataFrame) -> pd.DataFrame:
        data.AMPLITUDE = data.AMPLITUDE - data.AMPLITUDE.max()
        Theta_0 = data.ANGLE[data.AMPLITUDE.idxmax()]
        data.ANGLE = data.ANGLE - Theta_0
        return data

    def _normalize_PHASE(self, data: pd.DataFrame) -> pd.DataFrame:
        PHI_0 = data[data.ANGLE == 0]["PHASE"].to_numpy()[0]
        data["PHASE"] = np.unwrap(data.PHASE - PHI_0)
        return data

    def _get_taper_angle(self, data: pd.DataFrame) -> np.ndarray:
        data_interp = sp.interpolate.interp1d(data["ANGLE"], data["AMPLITUDE"])
        taper_angle = sp.optimize.minimize(
            lambda angle: np.abs(data_interp(angle) - self.taper),
            np.radians(10),
            method="nelder-mead",
        ).x[0]
        return taper_angle * np.ones(data["ANGLE"].shape)

    def _get_angle_phase(self, data: pd.DataFrame) -> np.ndarray:
        ph = sp.signal.savgol_filter(data.PHASE.values, 5, 2)
        peaks, _ = sp.signal.find_peaks(-ph, width=10)
        theta_peak = data.ANGLE.iloc[peaks].abs().min()
        return theta_peak * np.ones(data["ANGLE"].shape)

    def _load_dataset(self, file: str) -> pd.DataFrame:
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
                lambda data: data.assign(
                    FWHM=lambda data: self._get_taper_angle(data),
                ),
            )
            .reset_index()
            .groupby("FREQ")[["FREQ", "ANGLE", "PHASE", "AMPLITUDE", "FWHM"]]
            .apply(
                lambda data: data.assign(
                    THETA_CUT=lambda data: self._get_angle_phase(data),
                ),
            )
            .reset_index(drop=True)
            .groupby("FREQ")[
                ["FREQ", "ANGLE", "PHASE", "AMPLITUDE", "FWHM", "THETA_CUT"]
            ]
            .apply(
                lambda data: data.assign(
                    THETA_MAX=lambda data: np.min([data.FWHM, data.THETA_CUT]),
                ),
            )
            .query("ANGLE >= -THETA_MAX & ANGLE <= THETA_MAX")
            .reset_index(drop=True)
        )
        data_["DATASET"] = dataset

        data_ = (
            data_.groupby(["DATASET", "FREQ"])[data_.columns]
            .apply(
                lambda data: data.assign(
                    PHASE_SM=self._smooth_func(data.PHASE),
                ),
            )
            .reset_index(drop=True)
        )

        data_["s_AMP"] = self._sigma_Amp(data_.AMPLITUDE)
        data_["s_Uniform"] = self._sigma_Uniform(data_.AMPLITUDE)

        return data_

    def load_data(self) -> PhaseCenter:
        filenames = [self.path + file for file in self.dataset]
        return pd.concat(
            [self._load_dataset(filename) for filename in filenames],
        )

    @staticmethod
    def get_recent_file(path: str, mask: str) -> Path:
        files = list(Path(path).glob(mask))
        if not files:
            msg = (
                f"No files matching the mask '{mask}' found in the directory"
                "'{path}'."
            )
            raise ValueError(msg)
        return max(files, key=lambda file: file.stat().st_ctime)

    def load(self, path: str, mask: str | None = None) -> PhaseCenter:
        """Load the data from CSV files.

        Args:
        ----
            path (str): The path to the directory containing the CSV files.
            mask (str, optional): The mask to filter the CSV files. Defaults to
            None.

        Returns:
        -------
            PhaseCenter: The PhaseCenter object with loaded data.

        """
        masks = ["best_fit", "predicted", "data"]
        if mask is not None:
            _mask = [__mask + "*.csv" for __mask in masks]
            files = [PhaseCenter.get_recent_file(path, mask) for mask in _mask]
        else:
            _mask = [__mask + f"*{mask}*.csv" for __mask in masks]
            files = [PhaseCenter.get_recent_file(path, mask) for mask in _mask]
        self.best_fit = pd.read_csv(files[0])
        self.predicted = pd.read_csv(files[1])
        self.data = pd.read_csv(files[2])
        return self

    def save(
        self,
        suffix: str | None = None,
        path: str = "../data/processed",
    ) -> PhaseCenter:
        timestamp = pd.Timestamp.now().strftime("%d_%M_%Y_%H_%M_%S")
        self.best_fit.to_csv(
            f"{path}/best_fit_{timestamp}_{suffix}.csv",
            index=False,
        )
        self.predicted.to_csv(
            f"{path}/predicted_{timestamp}_{suffix}.csv",
            index=False,
        )
        self.data.to_csv(f"{path}/data_{timestamp}_{suffix}.csv", index=False)
        return self

    def _smooth_func(self, phase: np.ndarray) -> np.ndarray:
        return sp.signal.savgol_filter(phase.values, 20, 2)

    def _smooth_phases(self, data: pd.DataFrame) -> np.ndarray:
        phases = []
        for _, group in data.groupby(["DATASET", "FREQ"]):
            phases.append(self._smooth_func(group.PHASE))
        return np.concatenate(phases)

    def _fit_phase(
        self,
        model: callable,
        sigma_func: callable,
        phase_str: str,
        bootstrap: None | bool,
        cols_orig: list[str],
        par_names: list[str],
    ) -> pd.DataFrame:
        return (
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
                    ),
                ),
            )
            .reset_index()
            .rename(columns=dict(zip(cols_orig, par_names)))
        )

    def set_guess(self) -> PhaseCenter:
        self.guess = next(self._fit_phase())
        return self

    @staticmethod
    def _wrap_angle(theta: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(theta), np.cos(theta))

    @staticmethod
    def _k0(freq: float) -> float:
        return 2 * np.pi * freq * 1e9 / c

    @staticmethod
    def _sigma_Uniform(data: np.ndarray) -> np.ndarray:
        return np.ones(data.size)

    @staticmethod
    def _sigma_Amp(data: np.ndarray) -> np.ndarray:
        return 1 / (10 ** (data / 10))

    @staticmethod
    def model(
        theta: np.ndarray, DZ: float, PHI0: float, DXY: float
    ) -> np.ndarray:
        return DXY * np.sin(theta) + DZ * np.cos(theta) + PHI0

    @staticmethod
    def _fit_ols(
        data: np.ndarray,
        model: callable,
        guess: tuple[float, float, float],
        predict: bool = False,  # noqa: FBT002, FBT001
    ) -> tuple[float, float, float]:
        popt, pcov = sp.optimize.curve_fit(
            model,
            data[:, 0],
            data[:, 1],
            p0=guess,
            sigma=data[:, 2],
            maxfev=10000,
        )
        params = [*popt, *np.sqrt(np.diag(pcov))]
        if predict:
            y_pred = model(data[:, 0], *popt)
            params = [*params, *y_pred]
        return np.asarray(params)

    @staticmethod
    def _fit_odr(
        data: np.ndarray,
        model: callable,
        guess: tuple[float, float, float],
        predict: bool = False,  # noqa: FBT002, FBT001
    ) -> tuple[float, float, float]:
        model_odr = sp.odr.Model(model)
        DATA = sp.odr.RealData(
            data[:, 0],
            data[:, 1],
            sx=data[:, 2],
            sy=data[:, 3],
        )
        result = sp.odr.ODR(DATA, model_odr, beta0=guess).run()
        params = np.asarray(list(result.beta))
        perr = np.asarray(list(result.sd_beta))
        params = [*params, *perr]
        if predict:
            y_pred = result.y
            params = [*params, *y_pred]
        return np.asarray(params)

    @staticmethod
    def _fit_ols_gen(
        data: np.ndarray,
        model: callable,
        guess: tuple[float, float, float],
        predict: bool = False,  # noqa: FBT002, FBT001
    ) -> Generator[Any, Any, Any]:
        rng = np.random.default_rng()
        idx = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        data = data[idx]
        result = PhaseCenter._fit_ols(data, model, guess, predict=predict)
        yield np.asarray(result)

    @staticmethod
    def _fit_odr_gen(
        data: np.ndarray,
        model: callable,
        guess: tuple[float, float, float],
        predict: bool = False,  # noqa: FBT002, FBT001
    ) -> Generator[Any, Any, Any]:
        rng = np.random.default_rng()
        idx = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        data = data[idx]
        result = PhaseCenter._fit_odr(data, model, guess, predict=predict)
        yield np.asarray(result)

    @staticmethod
    def _fit(
        data: np.ndarray,
        model: Callable,
        guess: tuple[float, float, float],
        n: int = 100,
        predict: bool = False,  # noqa: FBT002, FBT001
    ) -> np.ndarray:
        EXPECTED_COLUMNS = 2

        if data.shape[1] == EXPECTED_COLUMNS:
            return np.asarray(
                [
                    next(
                        PhaseCenter._fit_ols_gen(
                            data,
                            model,
                            guess,
                            predict=predict,
                        ),
                    )
                    for x in np.arange(n)
                ],
            )
        return np.asarray(
            [
                next(
                    PhaseCenter._fit_ols_gen(
                        data,
                        model,
                        guess,
                        predict=predict,
                    ),
                )
                for x in np.arange(n)
            ],
        )

    @staticmethod
    def _bootstrap(
        data: np.ndarray, rng: np.random.Generator | None = None
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        n_pars = data.shape[1] // 2
        if data.shape[0] > 1:
            data = data[:, :n_pars]
            res = np.apply_along_axis(
                lambda data: sp.stats.bootstrap(
                    data.reshape(1, -1),
                    np.mean,
                    random_state=rng,
                ),
                0,
                data,
            )
            result = np.ravel(
                np.asarray(
                    [
                        [
                            np.mean(data[:, ii]),
                            np.abs(
                                np.mean(data[:, ii])
                                - rr.confidence_interval[0],
                            ),
                            np.abs(
                                np.mean(data[:, ii])
                                - rr.confidence_interval[1],
                            ),
                        ]
                        for ii, rr in enumerate(res)
                    ],
                ),
            )
        else:
            result = np.asarray(
                [
                    data[0, 0],
                    data[0, 3],
                    data[0, 3],
                    data[0, 1],
                    data[0, 4],
                    data[0, 4],
                    data[0, 2],
                    data[0, 5],
                    data[0, 5],
                ],
            )
        return result

    @staticmethod
    def _best_guess(
        data: np.ndarray,
        model: callable,
        guess: list[float],
        n: int,
    ) -> np.ndarray:
        return PhaseCenter._bootstrap(
            PhaseCenter._fit(data, model, guess, n=n),
        )

    def fit_phase_center(
        self,
        data: pd.DataFrame,
        model: callable,
        n: int = 50,
    ) -> list[float]:
        guess = [0.10, 0.10, 0.10]
        data = data.to_numpy()
        params = PhaseCenter._fit(data, model, guess, n=1)
        guess = params[0][:3]
        params = PhaseCenter._best_guess(data, model, guess, n=n)
        method = "OLS" if data.shape[1] == 3 else "ODR"
        result = np.concatenate([params, [method]])
        cols = [
            "DZ",
            "DZ_err_low",
            "DZ_err_high",
            "PHI_0",
            "PHI_0_err_low",
            "PHI_0_err_high",
            "DXY",
            "DXY_err_low",
            "DXY_err_high",
            "METHOD",
        ]
        return pd.DataFrame(result.reshape(1, -1), columns=cols)

    def _best_fit(self, cols: list[str], n: int) -> pd.DataFrame:
        result = self.data.groupby(["DATASET", "FREQ"])[
            [
                "ANGLE",
                *cols,
            ]
        ].apply(
            lambda data: self.fit_phase_center(
                data,
                PhaseCenter.model,
                n=n,
            ),
        )
        EXPECTED_COLUMNS = 2
        result["SMOOTH"] = "Smooth" if cols[0] == "PHASE_SM" else "Raw"
        result["WEIGHT"] = (
            cols[1] if len(cols) == EXPECTED_COLUMNS else "Uniform"
        )
        result["BOOTSTRAP"] = n
        return result

    def run_best_fit(self, n: int) -> PhaseCenter:
        weights = ["s_AMP", "s_Uniform"]
        phases = ["PHASE", "PHASE_SM"]
        if self.sigma_theta is not None:
            self.data["sigma_theta"] = self.sigma_theta
            self.data["sigma_phase"] = self.sigma_phase
            weights.append(["sigma_theta", "sigma_phase"])
        cols_sets = list(itertools.product(phases, weights))
        results = []
        for cols in cols_sets:
            _cols = list(cols)
            _cols = (
                _cols if isinstance(_cols[1], str) else [_cols[0], *_cols[1]]
            )
            result_1 = self._best_fit(_cols, 1)
            result_n = self._best_fit(_cols, n)
            result = pd.concat([result_1, result_n])
            results.append(result)
        self.best_fit = pd.concat(results).reset_index()
        cols = [col for col in self.best_fit.columns if "level" not in col]
        self.best_fit = self.best_fit[cols]
        float_cols = [
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
        self.best_fit[float_cols] = self.best_fit[float_cols].astype(float)
        return self

    def predict(self) -> np.ndarray:
        results = []
        data = self.data.copy()
        for name, group in self.best_fit.groupby(
            ["DATASET", "WEIGHT", "SMOOTH", "METHOD", "BOOTSTRAP", "FREQ"],
        ):
            guess = group[["DZ", "PHI_0", "DXY"]].to_numpy()[0].astype(float)
            data = self.data.query(
                "DATASET == @name[0] & FREQ == @name[-1]",
            )
            weight = (
                [name[1]]
                if name[1] != "Uniform"
                else ["sigma_theta", "sigma_phase"]
            )
            phase = ["PHASE_SM"] if name[2] == "Smooth" else ["PHASE"]
            cols = ["ANGLE", *phase, *weight]
            data = data[cols].copy().to_numpy()
            predicted = PhaseCenter.model(data[:, 0], *guess)

            result = pd.DataFrame(
                {
                    "ANGLE": data[:, 0],
                    "PHASE": data[:, 1],
                    "PREDICTED": predicted,
                },
            )
            result["DATASET"] = name[0]
            result["WEIGHT"] = name[1]
            result["SMOOTH"] = name[2]
            result["METHOD"] = name[3]
            result["BOOTSTRAP"] = name[4]
            result["FREQ"] = name[5]
            result["RESIDUALS"] = result.PHASE - result.PREDICTED
            results.append(result)

        self.predicted = pd.concat(results)
        return self

    @staticmethod
    def score_R2(data: pd.DataFrame) -> float:
        return 1 - np.sum((data.PHASE - data.PREDICTED) ** 2) / np.sum(
            (data.PHASE - data.PHASE.mean()) ** 2,
        )

    @staticmethod
    def score_Chi2(data: pd.DataFrame) -> float:
        return np.sum((data.PHASE - data.PREDICTED) ** 2 / (data.shape[0] - 3))

    @staticmethod
    def test_cramer(data: pd.DataFrame) -> float:
        try:
            res = sp.stats.cramervonmises_2samp(data.PHASE, data.PREDICTED)
        except ValueError:
            return np.nan
        return res.pvalue

    @staticmethod
    def test_KS(data: pd.DataFrame) -> float:
        return sp.stats.ks_2samp(data.PHASE, data.PREDICTED)[1]

    @staticmethod
    def test_KS_res(data: pd.DataFrame) -> float:
        return sp.stats.kstest(data.PHASE - data.PREDICTED, sp.stats.norm.cdf)[
            1
        ]

    def score(self) -> PhaseCenter:
        res = (
            self.predicted.groupby(self.GROUPS)
            .apply(
                lambda data: pd.Series(
                    {
                        "R2": PhaseCenter.score_R2(data),
                        "Chi2": PhaseCenter.score_Chi2(data),
                        "cramer": PhaseCenter.test_cramer(data),
                        "KS": PhaseCenter.test_KS(data),
                        "KS_res": PhaseCenter.test_KS_res(data),
                    },
                ),
            )
            .reset_index()
        )
        self.best_fit = self.best_fit.merge(
            res,
            on=self.GROUPS,
            how="inner",
        )
        return self

    def report(self) -> PhaseCenter:
        self.best_fit["Wavelength_cm"] = self.best_fit.apply(
            lambda data: np.round(PhaseCenter._wavelength_cm(data.FREQ), 2),
            axis=1,
        )
        self.best_fit["DZ_lambda"] = self.best_fit.apply(
            lambda data: PhaseCenter._DZ_lam(
                data.DZ,
                self.D0,
                data.FREQ,
            ),
            axis=1,
        )
        self.best_fit["DZ_err_low_lambda"] = self.best_fit.apply(
            lambda data: PhaseCenter._DZ_err_lam(
                data.DZ,
                data.DZ_err_low,
                data.FREQ,
                self.D0,
            ),
            axis=1,
        )
        self.best_fit["DZ_err_high_lambda"] = self.best_fit.apply(
            lambda data: PhaseCenter._DZ_err_lam(
                data.DZ,
                data.DZ_err_high,
                data.FREQ,
                self.D0,
            ),
            axis=1,
        )
        self.best_fit["DZ_phys"] = self.best_fit.apply(
            lambda data: PhaseCenter._DZ_phys(
                data.DZ,
                self.D0,
                data.FREQ,
            ),
            axis=1,
        )
        self.best_fit["DZ_err_low_phys"] = self.best_fit.apply(
            lambda data: data.DZ_err_low_lambda
            * PhaseCenter._wavelength_cm(data.FREQ),
            axis=1,
        )
        self.best_fit["DZ_err_high_phys"] = self.best_fit.apply(
            lambda data: data.DZ_err_high_lambda
            * PhaseCenter._wavelength_cm(data.FREQ),
            axis=1,
        )
        self.best_fit["DXY_lam"] = np.abs(self.best_fit["DXY"] / 2 / np.pi)
        self.best_fit["DXY_lam_err_low"] = (
            self.best_fit["DXY_err_low"] / 2 / np.pi
        )
        self.best_fit["DXY_lam_err_high"] = (
            self.best_fit["DZ_err_high"] / 2 / np.pi
        )

        self.best_fit["DXY_phys"] = (
            self.best_fit["DXY_lam"] * self.best_fit["Wavelength_cm"]
        )
        self.best_fit["DXY_err_low_phys"] = (
            self.best_fit["DXY_lam_err_low"] * self.best_fit["Wavelength_cm"]
        )
        self.best_fit["DXY_err_high_phys"] = (
            self.best_fit["DXY_lam_err_high"] * self.best_fit["Wavelength_cm"]
        )

        self.best_fit["PHI_0"] = np.degrees(self.best_fit["PHI_0"])
        self.best_fit["PHI_0_err_low"] = np.degrees(
            self.best_fit["PHI_0_err_low"],
        )
        self.best_fit["PHI_0_err_high"] = np.degrees(
            self.best_fit["PHI_0_err_high"],
        )
        self.best_fit["Wavelength_cm"] = np.round(
            self.best_fit["Wavelength_cm"],
            2,
        )
        return self

    def plot_phase_center(
        self,
        ax: plt.Axes | None = None,
        size: tuple[int, int] = (12, 4),
    ) -> plt.Axis:
        _data = self.best_fit.query(
            "WEIGHT == 's_Uniform' & SMOOTH == 'Smooth' & METHOD == 'OLS'",
        )
        colors = ["red", "blue"]
        styles = ["-", "--", "-.", ":"]
        if ax is None:
            _, ax = plt.subplots(nrows=1, ncols=2, figsize=size)
        __data = _data.sort_values(["DATASET", "BOOTSTRAP", "FREQ"])
        groups = list(__data.groupby(["DATASET", "BOOTSTRAP"]).groups.keys())
        for name, data in __data.groupby(["DATASET", "BOOTSTRAP"]):
            _ax1 = ["DZ", "DZ_err_low", "DZ_err_high"]
            _ax2 = ["DZ_phys", "DZ_err_low_phys", "DZ_err_high_phys"]
            _axs = [_ax1, _ax2]
            for ii, _ax in enumerate(_axs):
                y = data[_ax[0]]
                yerr = data[[_ax[1], _ax[2]]].T.to_numpy()
                ax[ii].errorbar(
                    1000 * data.FREQ,
                    y,
                    yerr=yerr,
                    color=colors[groups.index(name) // 2],
                    linestyle=styles[groups.index(name) // 2],
                    label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
                    fmt=".",
                    capsize=3,
                    linewidth=0.5,
                )
            ax[0].set_xlabel("Frequency (MHz)")
            ax[0].set_ylabel(r"Phase Center $\Delta_{Z}$ ($k_0$)")
            ax[1].set_xlabel("Frequency (MHz)")
            ax[1].set_ylabel(r"Phase Center $\Delta_{Z_\mathrm{{Phys}}}$ (cm)")
            ax[0].legend(
                loc="lower right",
                bbox_to_anchor=(1.5, -0.3),
                ncol=2,
            )
        return ax

    def plot_statistics(
        self,
        ax: plt.Axes = None,
        size: tuple[int, int] = (12, 8),
    ) -> plt.Axes:
        _data = self.best_fit.query(
            "WEIGHT == 's_Uniform' & SMOOTH == 'Smooth' & METHOD == 'OLS'",
        )
        __data = _data.sort_values(["DATASET", "BOOTSTRAP", "FREQ"])
        groups = list(__data.groupby(["DATASET", "BOOTSTRAP"]).groups.keys())
        colors = ["red", "blue"]
        styles = ["-", "--", "-.", ":"]
        if ax is None:
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=size)
        for name, data in _data.groupby(
            ["DATASET", "BOOTSTRAP"],
        ):
            ax[0, 0].plot(
                1000 * data.FREQ,
                data.R2,
                color=colors[groups.index(name) // 2],
                # linewidth=get_linewidth(name),
                linestyle=styles[groups.index(name) // 2],
                label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
            )
            ax[0, 1].plot(
                1000 * data.FREQ,
                data.cramer,
                color=colors[groups.index(name) // 2],
                # linewidth=get_linewidth(name),
                linestyle=styles[groups.index(name) // 2],
                label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
            )
            ax[1, 0].plot(
                1000 * data.FREQ,
                data.KS,
                color=colors[groups.index(name) // 2],
                # linewidth=get_linewidth(name),
                linestyle=styles[groups.index(name) // 2],
                label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
            )
            ax[1, 1].plot(
                1000 * data.FREQ,
                data.KS_res,
                color=colors[groups.index(name) // 2],
                # linewidth=get_linewidth(name),
                linestyle=styles[groups.index(name) // 2],
                label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
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
                color=colors[groups.index(name) // 2],
                # linewidth=get_linewidth(name),
                linestyle=styles[groups.index(name) // 2],
                label=f"Dataset: {name[0]}, Bootstrap: {name[1]}",
            )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = (
            functools.reduce(operator.iadd, lol, [])
            for lol in zip(*lines_labels)
        )

        # grab unique labels
        unique_labels = set(labels)

        # assign labels and legends in dict
        legend_dict = dict(zip(labels, lines))

        # query dict based on unique labels
        unique_lines = [legend_dict[x] for x in unique_labels]

        ax[2, 0].legend(
            unique_lines,
            unique_labels,
            loc="lower right",
            bbox_to_anchor=(1.8, -0.4),
            ncol=2,
        )
        return ax

    def plot_phases(
        self,
        freqs: list[float] | None = None,
        ncols: int = 3,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        nrows = int(np.ceil(2 * len(freqs) / ncols))
        size = (12, 3 * nrows)
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=size,
                gridspec_kw={"hspace": 0, "wspace": 0},
            )
        _data = self.predicted.query(
            "WEIGHT == 's_Uniform' and FREQ in @freqs",
        ).sort_values(["DATASET", "BOOTSTRAP", "FREQ", "ANGLE"])
        for name, _dataset in _data.groupby(
            ["DATASET", "BOOTSTRAP", "SMOOTH", "FREQ"],
        ):
            if name[0] == "Horizontal_Copolar":
                row = freqs.index(name[3]) // ncols
                col = freqs.index(name[3]) % ncols
                color = "red"
            else:
                row = (freqs.index(name[3]) + len(freqs)) // ncols
                col = (freqs.index(name[3]) + len(freqs)) % ncols
                color = "blue"

            meas = self.data.query("DATASET == @name[0] & FREQ == @name[3]")[
                ["ANGLE", "PHASE"]
            ]
            ax[row, col].scatter(
                np.degrees(meas.ANGLE),
                np.degrees(meas.PHASE),
                color="black",
                marker=".",
                alpha=0.5,
                label="Measured",
            )
            linewidth = 1 if name[1] == 500 else 0.5  # noqa: PLR2004
            linestyle = "solid" if name[2] == "Smooth" else "dotted"
            label = f"{name[0]} - {name[1]} - {name[2]}"
            ax[row, col].plot(
                np.degrees(_dataset.ANGLE),
                np.degrees(_dataset.PREDICTED),
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.5,
                label=label,
            )

        ax[0, 0].set_xlabel(
            r"$\theta$ (degrees)",
        )
        ax[nrows - 1, 0].set_ylabel(
            r"phase $\phi$ (degrees)",
        )

        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = (
            functools.reduce(operator.iadd, lol, [])
            for lol in zip(*lines_labels)
        )

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
            bbox_to_anchor=(2.3, -0.5),
            ncol=3,
            fontsize=6,
        )
        return ax

    def plot_residuals(
        self,
        freqs: list[float] | None = None,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        ncols = 4
        nrows = int(np.ceil(2 * len(freqs) / ncols))
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(12, 3 * nrows),
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
        _data = self.predicted.query(
            "WEIGHT == 's_Uniform' and FREQ in @freqs",
        ).sort_values(["DATASET", "BOOTSTRAP", "FREQ", "ANGLE"])
        groups = list(
            _data.groupby(
                ["DATASET", "BOOTSTRAP", "SMOOTH", "FREQ"]
            ).groups.keys(),
        )
        for name, _dataset in _data.groupby(
            ["DATASET", "BOOTSTRAP", "SMOOTH", "FREQ"],
        ):
            if name[0] == "Horizontal_Copolar":
                row = freqs.index(name[3]) // ncols
                col = freqs.index(name[3]) % ncols
            else:
                row = (freqs.index(name[3]) + len(freqs)) // ncols
                col = (freqs.index(name[3]) + len(freqs)) % ncols
            p1 = sm.ProbPlot(_dataset.PHASE)
            p2 = sm.ProbPlot(_dataset.PREDICTED)
            p1.qqplot(
                other=p2,
                line="45",
                xlabel="",
                ylabel="",
                markersize=2,
                marker=markers[groups.index(name) % len(markers)],
                alpha=alpha,
                label=f"Predicted - {name[0]}, {name[1]}, {name[2]}",
                ax=ax[row, col],
            )
            ax[row, col].set_title(
                f"Frequency: {1000 * name[3]} MHz", fontsize=6
            )

        ax[0, 0].set_xlabel(
            xlabel="Measured Phase",
        )
        ax[nrows - 1, 0].set_ylabel(
            ylabel="Predicted Phase",
        )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = (
            functools.reduce(operator.iadd, lol, [])
            for lol in zip(*lines_labels)
        )

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
            bbox_to_anchor=(2.5, -0.4),
            ncol=2,
            fontsize=6,
        )
        return ax

    def plot_qq_residuals(
        self,
        freqs: list[float] | None = None,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        ncols = 4
        nrows = int(np.ceil(2 * len(freqs) / ncols))
        if ax is None:
            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(12, 3 * nrows),
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
        _data = self.predicted.query(
            "WEIGHT == 's_Uniform' and FREQ in @freqs",
        ).sort_values(["DATASET", "BOOTSTRAP", "FREQ", "ANGLE"])
        groups = list(
            _data.groupby(
                ["DATASET", "BOOTSTRAP", "SMOOTH", "FREQ"]
            ).groups.keys(),
        )
        for name, _dataset in _data.groupby(
            ["DATASET", "BOOTSTRAP", "SMOOTH", "FREQ"],
        ):
            if name[0] == "Horizontal_Copolar":
                row = freqs.index(name[3]) // ncols
                col = freqs.index(name[3]) % ncols
            else:
                row = (freqs.index(name[3]) + len(freqs)) // ncols
                col = (freqs.index(name[3]) + len(freqs)) % ncols

            sm.qqplot(
                _dataset.RESIDUALS,
                line="s",
                xlabel="",
                ylabel="",
                markersize=2,
                marker=markers[groups.index(name) % len(markers)],
                alpha=alpha,
                label=f"Predicted - {name[0]}, {name[1]}, {name[2]}",
                ax=ax[row, col],
            )
            ax[row, col].set_title(
                f"Frequency: {1000 * name[3]} MHz", fontsize=6
            )

        ax[0, 0].set_xlabel(
            xlabel="Measured Phase",
        )
        ax[nrows - 1, 0].set_ylabel(
            ylabel="Predicted Phase",
        )
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = (
            functools.reduce(operator.iadd, lol, [])
            for lol in zip(*lines_labels)
        )

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
            bbox_to_anchor=(2.5, -0.4),
            ncol=2,
            fontsize=6,
        )
        return ax
