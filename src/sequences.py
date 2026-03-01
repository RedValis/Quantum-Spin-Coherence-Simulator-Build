"""
sequences.py - Pulse sequences for spin coherence simulation.
=============================================================

Prototype 4 implements:
  - apply_pulse          : instantaneous rotation via SO(3) rotation matrices
  - free_evolve          : Bloch ODE integration over a time segment
  - hahn_echo_sequence   : π/2 → τ → π → τ → echo
  - cpmg_sequence        : π/2 → [τ → π → τ]xN (multi-echo train)
  - sweep_echo_amplitude : echo amplitude vs 2τ (decay curve)
  - measure_echo_amplitude: |M_perp| at the echo time
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from .core import simulate_bloch, time_axis


# ===========================================================================
# Rotation matrices (SO(3))
# ===========================================================================

def _Rx(theta: float) -> np.ndarray:
    """3x3 rotation matrix around x-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def _Ry(theta: float) -> np.ndarray:
    """3x3 rotation matrix around y-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def _Rz(theta: float) -> np.ndarray:
    """3x3 rotation matrix around z-axis by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

_ROT = {
    'x':  _Rx,
    'y':  _Ry,
    'z':  _Rz,
    '-x': lambda t: _Rx(-t),
    '-y': lambda t: _Ry(-t),
    '-z': lambda t: _Rz(-t),
}


# ===========================================================================
# Core building blocks
# ===========================================================================

def apply_pulse(
    M: np.ndarray,
    axis: str = 'x',
    angle: float = np.pi / 2,
) -> np.ndarray:
    """Rotate Bloch vector M by *angle* around *axis* (instantaneous pulse).

    Implements the hard-pulse approximation: the pulse duration is negligible
    compared to T1, T2, and the Larmor period. The rotation is exact via
    SO(3) rotation matrices — zero numerical error in the pulse itself.

    Parameters
    ----------
    M     : (3,) array  Bloch vector [Mx, My, Mz]
    axis  : str         rotation axis — 'x', 'y', 'z', '-x', '-y', '-z'
    angle : float       rotation angle in radians
                          np.pi/2  →  90°  tip pulse
                          np.pi    →  180° inversion / refocusing pulse

    Returns
    -------
    M_rotated : (3,) np.ndarray

    Examples
    --------
    >>> apply_pulse(np.array([0., 0., 1.]), axis='y', angle=np.pi/2)
    array([1., 0., 0.])   # equilibrium tipped to +x
    >>> apply_pulse(np.array([0., 0., 1.]), axis='x', angle=np.pi)
    array([0., 0., -1.])  # full inversion

    Invariants
    ----------
    * |M_rotated| == |M|              (rotations preserve vector length)
    * apply_pulse(apply_pulse(M, ax, π), ax, π)  ==  M  (two π = identity)
    """
    M = np.asarray(M, dtype=float)
    if axis not in _ROT:
        raise ValueError(f"axis must be one of {list(_ROT)}, got '{axis}'")
    return _ROT[axis](angle) @ M


def free_evolve(
    M_init: np.ndarray,
    t_free: float,
    dt: float,
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    method: str = "RK45",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evolve M under free precession + T1/T2 relaxation for duration t_free.

    Thin wrapper around simulate_bloch that plays the role of a single
    *segment* in a larger pulse sequence. Returned time starts at 0
    (segment-relative); callers stitch absolute times themselves.

    Parameters
    ----------
    M_init  : (3,) array  initial state for this segment
    t_free  : float       segment duration
    dt      : float       time step
    gamma   : float       gyromagnetic ratio
    B       : (3,) array  field vector [Bx, By, Bz]
    T1, T2  : float       relaxation times
    M0      : float       equilibrium magnetisation
    method  : str         ODE solver passed to simulate_bloch

    Returns
    -------
    t_seg, Mx, My, Mz : np.ndarray, shape (N,) each
    """
    return simulate_bloch(
        M_init=M_init,
        gamma=gamma, B=B,
        T1=T1, T2=T2, M0=M0,
        t_max=t_free, dt=dt,
        method=method,
    )


# ===========================================================================
# Hahn echo
# ===========================================================================

def hahn_echo_sequence(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau: float,
    dt: float,
    tip_axis: str = 'y',
    refocus_axis: str = 'x',
    M_eq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Hahn spin-echo sequence.

    Sequence
    --------
    1. π/2 pulse (tip_axis)    - tips equilibrium spin into transverse plane
    2. Free evolution τ         - spins dephase (T2 decay + precession)
    3. π pulse (refocus_axis)  - reverses phase accumulated in step 2
    4. Free evolution τ         - spins rephase → echo forms at t = 2τ

    For a single homogeneous spin the echo amplitude is:

        |M_echo| = M0 · exp(−2τ / T2)

    The π pulse refocuses *static* field inhomogeneity (addressed in
    Prototype 5 — ensemble). For a single spin it correctly reproduces
    the T2 envelope.

    Parameters
    ----------
    gamma        : float       gyromagnetic ratio
    B            : (3,) array  field [Bx, By, Bz], typically [0, 0, B0]
    T1, T2       : float       relaxation times (T2 ≤ T1)
    M0           : float       equilibrium magnetisation
    tau          : float       half-echo time; echo appears at t = 2τ
    dt           : float       time step (must be < tau)
    tip_axis     : str         axis for π/2 tip pulse (default 'y' → tips to +x)
    refocus_axis : str         axis for π refocusing pulse (default 'x')
    M_eq         : (3,) array  state before π/2 (default [0, 0, M0])

    Returns
    -------
    t, Mx, My, Mz : np.ndarray, shape (N,) each
        Concatenated time axis 0 → 2τ and Bloch components.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt >= tau:
        raise ValueError(f"dt ({dt}) must be smaller than tau ({tau})")

    if M_eq is None:
        M_eq = np.array([0.0, 0.0, M0])
    else:
        M_eq = np.asarray(M_eq, dtype=float)

    # -- 1. π/2 tip pulse (instantaneous) ------------------------------------
    M1 = apply_pulse(M_eq, axis=tip_axis, angle=np.pi / 2)

    # -- 2. Free evolution for τ ----------------------------------------------
    t1, Mx1, My1, Mz1 = free_evolve(M1, tau, dt, gamma, B, T1, T2, M0)

    # -- 3. π refocusing pulse ------------------------------------------------
    M2 = apply_pulse(np.array([Mx1[-1], My1[-1], Mz1[-1]]),
                     axis=refocus_axis, angle=np.pi)

    # -- 4. Free evolution for τ (echo forms at end) --------------------------
    t2, Mx2, My2, Mz2 = free_evolve(M2, tau, dt, gamma, B, T1, T2, M0)

    # -- Stitch (drop the duplicate boundary point) ---------------------------
    t_full = np.concatenate([t1, t1[-1] + t2[1:]])
    Mx     = np.concatenate([Mx1, Mx2[1:]])
    My     = np.concatenate([My1, My2[1:]])
    Mz     = np.concatenate([Mz1, Mz2[1:]])

    return t_full, Mx, My, Mz


# ===========================================================================
# CPMG multi-echo train
# ===========================================================================

def cpmg_sequence(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau: float,
    n_echoes: int,
    dt: float,
    tip_axis: str = 'y',
    refocus_axis: str = 'x',
    M_eq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a CPMG multi-echo train.

    Sequence
    --------
    π/2  →  [τ → π → τ → (echo)]xn_echoes

    Echo amplitudes decay as exp(−t_echo / T2) where t_echo = 2k·τ.
    CPMG is robust to pulse imperfections because the Meiboom-Gill phase
    (tip on y, refocus on x) corrects systematic errors.

    Parameters
    ----------
    gamma, B, T1, T2, M0 : as in hahn_echo_sequence
    tau      : float   half-spacing between π pulses
    n_echoes : int     number of refocusing π pulses (= number of echoes)
    dt       : float   time step

    Returns
    -------
    t, Mx, My, Mz : full time axis and Bloch components
    echo_times    : (n_echoes,) array — echo centres at 2k·τ, k=1..n_echoes
    """
    if n_echoes < 1:
        raise ValueError(f"n_echoes must be ≥ 1, got {n_echoes}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt >= tau:
        raise ValueError(f"dt ({dt}) must be smaller than tau ({tau})")

    if M_eq is None:
        M_eq = np.array([0.0, 0.0, M0])
    else:
        M_eq = np.asarray(M_eq, dtype=float)

    # π/2 tip
    M_cur = apply_pulse(M_eq, axis=tip_axis, angle=np.pi / 2)

    t_all  = np.array([0.0])
    Mx_all = np.array([M_cur[0]])
    My_all = np.array([M_cur[1]])
    Mz_all = np.array([M_cur[2]])
    t_offset   = 0.0
    echo_times = []

    for _ in range(n_echoes):
        # free evolve τ before π
        t_s, Mx_s, My_s, Mz_s = free_evolve(
            M_cur, tau, dt, gamma, B, T1, T2, M0)
        M_cur    = np.array([Mx_s[-1], My_s[-1], Mz_s[-1]])
        t_all    = np.concatenate([t_all,  t_offset + t_s[1:]])
        Mx_all   = np.concatenate([Mx_all, Mx_s[1:]])
        My_all   = np.concatenate([My_all, My_s[1:]])
        Mz_all   = np.concatenate([Mz_all, Mz_s[1:]])
        t_offset += t_s[-1]

        # π refocusing pulse
        M_cur = apply_pulse(M_cur, axis=refocus_axis, angle=np.pi)

        # free evolve τ after π → echo at end of this segment
        t_s2, Mx_s2, My_s2, Mz_s2 = free_evolve(
            M_cur, tau, dt, gamma, B, T1, T2, M0)
        M_cur    = np.array([Mx_s2[-1], My_s2[-1], Mz_s2[-1]])
        t_all    = np.concatenate([t_all,  t_offset + t_s2[1:]])
        Mx_all   = np.concatenate([Mx_all, Mx_s2[1:]])
        My_all   = np.concatenate([My_all, My_s2[1:]])
        Mz_all   = np.concatenate([Mz_all, Mz_s2[1:]])
        t_offset += t_s2[-1]

        echo_times.append(t_offset)

    return t_all, Mx_all, My_all, Mz_all, np.array(echo_times)


# ===========================================================================
# Echo amplitude measurement
# ===========================================================================

def measure_echo_amplitude(
    Mx: np.ndarray,
    My: np.ndarray,
    t: np.ndarray,
    echo_time: float,
) -> float:
    """Return transverse magnitude |M_perp| at the echo time.

    Finds the index in *t* closest to *echo_time* and returns
    sqrt(Mx² + My²) at that point.

    For a single homogeneous spin, |M_perp| decays monotonically —
    there is no local maximum to search for. The echo amplitude is
    simply the value at t = echo_time:

        |M_echo| = M0 · exp(−2τ / T2)

    For inhomogeneous ensembles (Prototype 5), a true peak forms at
    echo_time and this function will correctly return it.

    Parameters
    ----------
    Mx, My     : component arrays
    t          : time axis (same length as Mx, My)
    echo_time  : expected echo time (e.g. 2τ for Hahn echo)

    Returns
    -------
    float  |M_perp| at the closest sampled time to echo_time
    """
    M_perp = np.sqrt(Mx**2 + My**2)
    idx    = np.argmin(np.abs(t - echo_time))
    return float(M_perp[idx])


def sweep_echo_amplitude(
    gamma: float,
    B: np.ndarray,
    T1: float,
    T2: float,
    M0: float,
    tau_values: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep τ and record Hahn echo amplitude at t = 2τ for each value.

    Generates the canonical T2 decay curve:

        A_echo(2τ) = M0 · exp(−2τ / T2)

    Parameters
    ----------
    gamma, B, T1, T2, M0 : physics parameters
    tau_values : (N,) array of τ values to sweep over
    dt         : time step for each Hahn echo simulation

    Returns
    -------
    two_tau    : (N,) array   echo times = 2·tau_values
    amplitudes : (N,) array   measured echo amplitude at each echo time
    """
    amplitudes = []
    for tau in tau_values:
        t, Mx, My, _ = hahn_echo_sequence(
            gamma=gamma, B=B, T1=T1, T2=T2, M0=M0,
            tau=float(tau), dt=dt)
        amp = measure_echo_amplitude(Mx, My, t, echo_time=2.0 * float(tau))
        amplitudes.append(amp)
    return 2.0 * np.asarray(tau_values), np.array(amplitudes)