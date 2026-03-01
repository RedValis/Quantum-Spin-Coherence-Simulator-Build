"""
fitting.py - Parameter extraction from experimental coherence data.
===================================================================

Prototype 6 provides:
  - simulate_coherence_for_fit : scipy curve_fit-compatible wrapper
  - fit_T2_to_data             : single-parameter T2 fit
  - fit_multi_to_data          : multi-parameter fit (T2 + sigma, or T1 + T2)
  - generate_synthetic_data    : make noisy test data for UI demo
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import curve_fit


# ===========================================================================
# Model wrappers (curve_fit requires signature f(t, *params))
# ===========================================================================

def _model_simple_T2(t: np.ndarray, T2: float) -> np.ndarray:
    """L(t) = exp(-t/T2)."""
    return np.exp(-t / T2)


def _model_gaussian_fid(t: np.ndarray, T2: float, sigma: float) -> np.ndarray:
    """Ensemble FID envelope: exp(-t/T2) * exp(-sigma^2 * t^2 / 2)."""
    return np.exp(-t / T2) * np.exp(-0.5 * sigma**2 * t**2)


def _model_hahn_echo(t_half: np.ndarray, T2: float) -> np.ndarray:
    """Echo amplitude vs tau: exp(-2*tau/T2).
    t_half is the tau array (half echo time); echo time = 2*tau.
    """
    return np.exp(-2 * t_half / T2)


def _model_T1_recovery(t: np.ndarray, T1: float, M0: float) -> np.ndarray:
    """Mz recovery: M0 * (1 - exp(-t/T1))."""
    return M0 * (1 - np.exp(-t / T1))


def _model_stretched_T2(t: np.ndarray, T2: float, beta: float) -> np.ndarray:
    """Stretched exponential: exp(-(t/T2)^beta). beta=1 is pure T2."""
    return np.exp(-(t / T2) ** beta)


MODEL_REGISTRY = {
    "simple_T2":       (_model_simple_T2,       ["T2"],          [5.0]),
    "gaussian_fid":    (_model_gaussian_fid,     ["T2", "sigma"], [5.0, 0.2]),
    "hahn_echo":       (_model_hahn_echo,        ["T2"],          [5.0]),
    "T1_recovery":     (_model_T1_recovery,      ["T1", "M0"],    [20.0, 1.0]),
    "stretched_T2":    (_model_stretched_T2,     ["T2", "beta"],  [5.0, 1.0]),
}


def simulate_coherence_for_fit(
    t: np.ndarray,
    T2: float,
    model: str = "simple_T2",
) -> np.ndarray:
    """Wrapper used by curve_fit: return simulated coherence for given t and T2.

    For the simple and echo models (single parameter), this is the direct
    curve_fit-compatible function.

    Parameters
    ----------
    t     : time array
    T2    : transverse relaxation time to evaluate
    model : one of 'simple_T2', 'hahn_echo'

    Returns
    -------
    Coherence values L(t)
    """
    if model == "simple_T2":
        return _model_simple_T2(t, T2)
    elif model == "hahn_echo":
        return _model_hahn_echo(t, T2)
    else:
        raise ValueError(
            f"simulate_coherence_for_fit only supports single-param models. "
            f"Use fit_multi_to_data for '{model}'."
        )


def fit_T2_to_data(
    t_data: np.ndarray,
    L_data: np.ndarray,
    initial_guess: float = 5.0,
    model: str = "simple_T2",
    bounds: Tuple = (0.01, 1e6),
) -> Tuple[float, float, np.ndarray]:
    """Fit T2 to experimental coherence data using scipy curve_fit.

    Parameters
    ----------
    t_data        : time axis of experimental data
    L_data        : measured coherence values (normalised to [0, 1])
    initial_guess : starting T2 value for optimiser
    model         : 'simple_T2' or 'hahn_echo'
    bounds        : (lower, upper) bound on T2

    Returns
    -------
    T2_fit    : float   best-fit T2
    T2_stderr : float   1-sigma uncertainty from covariance matrix
    L_fit     : (N,)    fitted curve evaluated at t_data
    """
    t_data = np.asarray(t_data, dtype=float)
    L_data = np.asarray(L_data, dtype=float)

    def _model(t, T2):
        return simulate_coherence_for_fit(t, T2, model=model)

    popt, pcov = curve_fit(
        _model, t_data, L_data,
        p0=[initial_guess],
        bounds=([bounds[0]], [bounds[1]]),
        maxfev=10000,
    )
    T2_fit    = float(popt[0])
    T2_stderr = float(np.sqrt(np.diag(pcov))[0])
    L_fit     = _model(t_data, T2_fit)
    return T2_fit, T2_stderr, L_fit


def fit_multi_to_data(
    t_data: np.ndarray,
    L_data: np.ndarray,
    model: str = "gaussian_fid",
    initial_guess: Optional[list] = None,
    bounds_lower: Optional[list] = None,
    bounds_upper: Optional[list] = None,
) -> Tuple[dict, dict, np.ndarray]:
    """Fit multiple parameters to data using scipy curve_fit.

    Parameters
    ----------
    t_data        : time axis
    L_data        : measured coherence
    model         : key in MODEL_REGISTRY
    initial_guess : list of starting values (uses registry defaults if None)
    bounds_lower  : lower bounds for each parameter
    bounds_upper  : upper bounds for each parameter

    Returns
    -------
    params     : dict  {param_name: best_fit_value}
    errors     : dict  {param_name: 1-sigma_uncertainty}
    L_fit      : (N,)  fitted curve at t_data
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(MODEL_REGISTRY)}")

    fn, param_names, defaults = MODEL_REGISTRY[model]
    p0     = initial_guess if initial_guess is not None else defaults
    lb     = bounds_lower  if bounds_lower  is not None else [1e-4] * len(param_names)
    ub     = bounds_upper  if bounds_upper  is not None else [1e6]  * len(param_names)

    t_data = np.asarray(t_data, dtype=float)
    L_data = np.asarray(L_data, dtype=float)

    popt, pcov = curve_fit(
        fn, t_data, L_data,
        p0=p0,
        bounds=(lb, ub),
        maxfev=20000,
    )
    stds   = np.sqrt(np.diag(pcov))
    params = {name: float(v) for name, v in zip(param_names, popt)}
    errors = {name: float(s) for name, s in zip(param_names, stds)}
    L_fit  = fn(t_data, *popt)
    return params, errors, L_fit


def generate_synthetic_data(
    t_max: float,
    n_points: int,
    T2: float,
    noise_level: float = 0.03,
    model: str = "simple_T2",
    seed: int = 0,
    **extra_params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate noisy synthetic data for testing the fitting pipeline.

    Parameters
    ----------
    t_max       : maximum time
    n_points    : number of data points
    T2          : true T2 to embed in synthetic data
    noise_level : std of Gaussian noise added
    model       : which model to use
    seed        : RNG seed

    Returns
    -------
    t_data, L_noisy : time axis and noisy coherence values
    """
    rng    = np.random.default_rng(seed)
    t_data = np.linspace(0, t_max, n_points)

    if model == "simple_T2":
        L_clean = _model_simple_T2(t_data, T2)
    elif model == "gaussian_fid":
        sigma = extra_params.get("sigma", 0.2)
        L_clean = _model_gaussian_fid(t_data, T2, sigma)
    elif model == "hahn_echo":
        L_clean = _model_hahn_echo(t_data, T2)
    elif model == "T1_recovery":
        M0 = extra_params.get("M0", 1.0)
        T1 = extra_params.get("T1", T2 * 3)
        L_clean = _model_T1_recovery(t_data, T1, M0)
    else:
        L_clean = _model_simple_T2(t_data, T2)

    L_noisy = L_clean + rng.normal(0, noise_level, len(t_data))
    L_noisy = np.clip(L_noisy, -0.1, 1.1)
    return t_data, L_noisy