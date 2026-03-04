"""
density_matrix.py - Open quantum system evolution via density matrix.
======================================================================

Moves from the Bloch vector (a classical 3-vector) to the density matrix
(a 2×2 complex Hermitian operator), which is the proper quantum mechanical
description of a spin-½ system.

State representation
--------------------
A pure state |ψ⟩ = α|↑⟩ + β|↓⟩ has density matrix:

    ρ = |ψ⟩⟨ψ| = [[|α|², αβ*],
                   [α*β,  |β|²]]

A mixed state is a classical probability mixture of pure states:

    ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|

The Bloch vector r⃗ is recovered from ρ via the Pauli matrices:

    rx = 2·Re(ρ₀₁) = Tr(ρ·σx)
    ry = 2·Im(ρ₀₁) = Tr(ρ·σy)
    rz = ρ₀₀ - ρ₁₁ = Tr(ρ·σz)

|r⃗| = 1 ↔ pure state;  |r⃗| < 1 ↔ mixed state.

Lindblad master equation (Step 5)
-----------------------------------
The full open-system evolution under arbitrary Markovian noise:

    dρ/dt = -(i/ℏ)[H, ρ] + Σ_k D[L_k]ρ

where each Lindblad dissipator is:

    D[L]ρ = L ρ L† - ½{L†L, ρ}

Three canonical noise channels are implemented:

  1. Amplitude Damping (T₁)
     Models energy relaxation: |↓⟩ → |↑⟩ spontaneous emission.
     L = √γ₁ · σ₋,   γ₁ = 1/T₁
     Effect: ρ₁₁ → 0 (excited population decays), ρ₀₁ → ρ₀₁·exp(-t/2T₁)
     Bloch: Mz → +1 (ground state), |M_perp| decays at half rate of Mz.

  2. Phase Damping / Pure Dephasing (T₂φ)
     Models random phase kicks with no energy exchange.
     L = √γ_φ · σz,  γ_φ = 1/(2T₂) - 1/(4T₁)   (pure dephasing contribution)
     Effect: off-diagonal ρ₀₁ → 0, populations unchanged.
     Bloch: M_perp → 0, Mz unaffected.

  3. Depolarizing Noise (T_dep)
     Models isotropic noise — errors equally likely in all directions.
     L_x = √(γ_dep/4) · σx
     L_y = √(γ_dep/4) · σy
     L_z = √(γ_dep/4) · σz,   γ_dep = 1/T_dep
     Effect: any state → I/2 (maximally mixed) at rate γ_dep.
     Bloch: r⃗ → 0 isotropically.  Not physical for real qubits but a
            common model for benchmarking error rates.

Combined T₁/T₂ rates:
  Total dephasing:  1/T₂ = 1/(2T₁) + γ_φ
  Depolarizing adds:  γ_dep contracts r⃗ uniformly.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field


# ============================================================================
# Pauli matrices and basis states
# ============================================================================

I2      = np.eye(2, dtype=complex)
sx      = np.array([[0, 1],   [1,  0]],  dtype=complex)
sy      = np.array([[0, -1j], [1j, 0]],  dtype=complex)
sz      = np.array([[1, 0],   [0, -1]],  dtype=complex)
s_plus  = np.array([[0, 1],   [0,  0]],  dtype=complex)   # |↑⟩⟨↓| — raises
s_minus = np.array([[0, 0],   [1,  0]],  dtype=complex)   # |↓⟩⟨↑| — lowers

KET_UP   = np.array([[1], [0]], dtype=complex)   # |↑⟩ — ground state
KET_DOWN = np.array([[0], [1]], dtype=complex)   # |↓⟩ — excited state


# ============================================================================
# Noise channel configuration
# ============================================================================

@dataclass
class NoiseModel:
    """
    Container for noise parameters selecting which channels are active.

    Each channel can be independently enabled or disabled by setting its
    rate parameter.  All rates are in 1/time_unit.

    Parameters
    ----------
    T1      : Longitudinal relaxation time (amplitude damping).
              Set to np.inf to disable amplitude damping.
    T2      : Total transverse relaxation time (combined T1 + pure dephasing).
              Must satisfy T2 ≤ T1.  Set T2 = T1 * 2 (maximum) for pure T1.
    T_dep   : Depolarizing time.  Set to np.inf to disable.
    channels: List of active channel names.  Populated automatically.

    Examples
    --------
    # Amplitude damping only (T2 = 2·T1 → zero pure dephasing):
    nm = NoiseModel(T1=10.0, T2=20.0, T_dep=np.inf)

    # Pure dephasing only (no energy relaxation, fast T2):
    nm = NoiseModel(T1=np.inf, T2=5.0, T_dep=np.inf)

    # All three channels active:
    nm = NoiseModel(T1=10.0, T2=8.0, T_dep=30.0)

    # Depolarizing only:
    nm = NoiseModel(T1=np.inf, T2=np.inf, T_dep=15.0)
    """
    T1:    float = np.inf
    T2:    float = np.inf
    T_dep: float = np.inf
    channels: List[str] = field(default_factory=list)

    def __post_init__(self):
        if np.isfinite(self.T1) and np.isfinite(self.T2):
            # Effective T2 cannot exceed 2*T1 for physical reasons
            T2_max = 2.0 * self.T1
            if self.T2 > T2_max:
                raise ValueError(
                    f"T2 ({self.T2}) cannot exceed 2·T1 ({T2_max}) — "
                    "pure dephasing rate would be negative (unphysical)."
                )
        active = []
        if np.isfinite(self.T1):
            active.append("amplitude_damping")
        if np.isfinite(self.T2) and self._gamma_phi() > 0:
            active.append("phase_damping")
        if np.isfinite(self.T_dep):
            active.append("depolarizing")
        self.channels = active

    def _gamma1(self) -> float:
        """Amplitude damping rate γ₁ = 1/T₁."""
        return 1.0 / self.T1 if np.isfinite(self.T1) else 0.0

    def _gamma_phi(self) -> float:
        """Pure dephasing rate γ_φ = 1/(2T₂) - 1/(4T₁), clamped ≥ 0."""
        rate = 0.0
        if np.isfinite(self.T2):
            rate += 1.0 / (2.0 * self.T2)
        if np.isfinite(self.T1):
            rate -= 1.0 / (4.0 * self.T1)
        return max(0.0, rate)

    def _gamma_dep(self) -> float:
        """Depolarizing rate γ_dep = 1/T_dep."""
        return 1.0 / self.T_dep if np.isfinite(self.T_dep) else 0.0

    def lindblad_operators(self) -> List[Tuple[np.ndarray, float]]:
        """
        Return list of (operator, rate) pairs for all active channels.

        Each pair (L, γ) contributes γ · D[L]ρ to the master equation.
        The rate is pre-absorbed for efficiency; the dissipator uses
        the un-scaled operator L directly.

        Returns
        -------
        ops : list of (2×2 complex ndarray, float rate)

        Note on amplitude damping operator
        ------------------------------------
        The jump operator for energy relaxation |↓⟩ → |↑⟩ is:
            L = σ₊ = |↑⟩⟨↓| = [[0,1],[0,0]]   (s_plus in this module)

        This is the operator that maps the excited state |↓⟩ (south pole,
        ρ₁₁) to the ground state |↑⟩ (north pole, ρ₀₀).  Applying D[σ₊]:
            dρ₀₀/dt = +γ₁·ρ₁₁   (ground population grows)
            dρ₁₁/dt = -γ₁·ρ₁₁   (excited population decays to 0)
            dρ₀₁/dt = -γ₁/2·ρ₀₁ (coherence decays at half rate)
        which matches the Bloch-equation T₁ term: dMz/dt = -(Mz-1)/T₁.
        Using s_minus instead would cause excitation (drive ground → excited),
        which is unphysical for spontaneous emission.
        """
        ops = []

        # 1. Amplitude damping: L = s_plus = |↑⟩⟨↓|  (excited → ground decay)
        γ1 = self._gamma1()
        if γ1 > 0:
            ops.append((s_plus, γ1))

        # 2. Pure phase damping (dephasing)
        γφ = self._gamma_phi()
        if γφ > 0:
            ops.append((sz, γφ))

        # 3. Depolarizing — three equal-strength channels
        γd = self._gamma_dep()
        if γd > 0:
            # D[σx] + D[σy] + D[σz] drives ρ → I/2 at rate γ_dep
            for op in (sx, sy, sz):
                ops.append((op, γd / 4.0))

        return ops

    def effective_T2_star(self) -> float:
        """
        Effective transverse coherence time including all noise sources.

        1/T₂* = 1/(2T₁) + γ_φ + γ_dep
        """
        rate = 0.0
        if np.isfinite(self.T1):
            rate += 1.0 / (2.0 * self.T1)
        rate += self._gamma_phi()
        rate += self._gamma_dep()
        return 1.0 / rate if rate > 0 else np.inf

    def effective_T1_star(self) -> float:
        """
        Effective longitudinal relaxation time including depolarizing.

        1/T₁* = 1/T₁ + γ_dep
        """
        rate = self._gamma1() + self._gamma_dep()
        return 1.0 / rate if rate > 0 else np.inf

    def summary(self) -> str:
        """Human-readable summary of active channels and rates."""
        lines = ["NoiseModel summary", "=" * 40]
        lines.append(f"  T1        = {self.T1:.4g}")
        lines.append(f"  T2        = {self.T2:.4g}")
        lines.append(f"  T_dep     = {self.T_dep:.4g}")
        lines.append(f"  γ₁        = {self._gamma1():.4g}  (amplitude damping)")
        lines.append(f"  γ_φ       = {self._gamma_phi():.4g}  (pure dephasing)")
        lines.append(f"  γ_dep     = {self._gamma_dep():.4g}  (depolarizing)")
        lines.append(f"  T₂*_eff   = {self.effective_T2_star():.4g}")
        lines.append(f"  T₁*_eff   = {self.effective_T1_star():.4g}")
        lines.append(f"  Active channels: {self.channels}")
        return "\n".join(lines)


# ============================================================================
# Convenience constructors for standard noise models
# ============================================================================

def noise_amplitude_damping(T1: float) -> NoiseModel:
    """Pure amplitude damping (no pure dephasing, no depolarizing).

    Sets T2 = 2·T1 so that γ_φ = 0 exactly.
    Physical model for spontaneous emission from an excited state.

    Parameters
    ----------
    T1 : longitudinal relaxation time

    Returns
    -------
    NoiseModel with only amplitude damping active
    """
    return NoiseModel(T1=T1, T2=2.0 * T1, T_dep=np.inf)


def noise_phase_damping(T2: float) -> NoiseModel:
    """Pure phase damping — dephasing with no energy relaxation.

    T1 → ∞ so there is no amplitude damping contribution.
    Physical model for low-frequency magnetic field fluctuations.

    Parameters
    ----------
    T2 : transverse coherence time (all from pure dephasing)

    Returns
    -------
    NoiseModel with only phase damping active
    """
    return NoiseModel(T1=np.inf, T2=T2, T_dep=np.inf)


def noise_depolarizing(T_dep: float) -> NoiseModel:
    """Pure depolarizing channel — isotropic noise in all directions.

    Drives any state to I/2 (maximally mixed) at rate 1/T_dep.
    Common benchmarking model; not directly physical but parametrises
    gate error rates in quantum computing.

    Parameters
    ----------
    T_dep : depolarizing time constant

    Returns
    -------
    NoiseModel with only depolarizing active
    """
    return NoiseModel(T1=np.inf, T2=np.inf, T_dep=T_dep)


def noise_combined(T1: float, T2: float, T_dep: float = np.inf) -> NoiseModel:
    """All noise channels active simultaneously.

    Parameters
    ----------
    T1    : longitudinal relaxation time  (T1 ≤ ∞)
    T2    : total transverse time         (T2 ≤ 2·T1)
    T_dep : depolarizing time             (default: np.inf = off)

    Returns
    -------
    NoiseModel with amplitude damping + pure dephasing [+ depolarizing]
    """
    return NoiseModel(T1=T1, T2=T2, T_dep=T_dep)


# ============================================================================
# State constructors
# ============================================================================

def ground_state_dm() -> np.ndarray:
    """Density matrix for |↑⟩ (ground state, Mz = +1)."""
    return np.array([[1, 0], [0, 0]], dtype=complex)


def excited_state_dm() -> np.ndarray:
    """Density matrix for |↓⟩ (excited state, Mz = -1)."""
    return np.array([[0, 0], [0, 1]], dtype=complex)


def superposition_dm(theta: float = np.pi / 2, phi: float = 0.0) -> np.ndarray:
    """Density matrix for a pure state on the Bloch sphere.

    |ψ⟩ = cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩

    Parameters
    ----------
    theta : polar angle from north pole (0 = |↑⟩, π = |↓⟩, π/2 = equator)
    phi   : azimuthal angle (0 = +x, π/2 = +y)

    Returns
    -------
    rho : (2,2) complex density matrix
    """
    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)
    psi   = np.array([[alpha], [beta]], dtype=complex)
    return psi @ psi.conj().T


def mixed_state_dm(purity: float) -> np.ndarray:
    """Maximally mixed state scaled by purity p: ρ = p·|+x⟩⟨+x| + (1-p)·I/2.

    purity = 1.0 → pure |+x⟩ state  (Bloch vector on equator)
    purity = 0.0 → completely mixed  (Bloch vector at origin)
    """
    if not (0 <= purity <= 1):
        raise ValueError(f"purity must be in [0, 1], got {purity}")
    rho_pure = superposition_dm(theta=np.pi / 2, phi=0.0)
    return purity * rho_pure + (1 - purity) * I2 / 2


# ============================================================================
# Bloch vector ↔ density matrix conversions
# ============================================================================

def dm_to_bloch(rho: np.ndarray) -> np.ndarray:
    """Extract Bloch vector r⃗ = (rx, ry, rz) from density matrix ρ."""
    rx = float(np.real(np.trace(rho @ sx)))
    ry = float(np.real(np.trace(rho @ sy)))
    rz = float(np.real(np.trace(rho @ sz)))
    return np.array([rx, ry, rz])


def bloch_to_dm(r: np.ndarray) -> np.ndarray:
    """Construct density matrix from Bloch vector r⃗ = (rx, ry, rz).

    ρ = (I + r⃗·σ⃗) / 2
    """
    rx, ry, rz = r
    return (I2 + rx * sx + ry * sy + rz * sz) / 2


def purity(rho: np.ndarray) -> float:
    """Return Tr(ρ²) — equals 1 for pure states, 0.5 for maximally mixed."""
    return float(np.real(np.trace(rho @ rho)))


def population(rho: np.ndarray) -> Tuple[float, float]:
    """Return (P_ground, P_excited) = (ρ₀₀, ρ₁₁)."""
    return float(np.real(rho[0, 0])), float(np.real(rho[1, 1]))


def coherence(rho: np.ndarray) -> complex:
    """Return off-diagonal element ρ₀₁ — the quantum coherence."""
    return complex(rho[0, 1])


# ============================================================================
# Hamiltonian builders
# ============================================================================

def hamiltonian_free(omega0: float) -> np.ndarray:
    """Free precession: H = (ω₀/2)·σz."""
    return (omega0 / 2) * sz


def hamiltonian_rotating(delta: float, omega_rabi: float) -> np.ndarray:
    """Rotating-frame RWA Hamiltonian: H = (Δ/2)·σz + (Ω/2)·σx."""
    return (delta / 2) * sz + (omega_rabi / 2) * sx


def hamiltonian_free_evolution(delta: float) -> np.ndarray:
    """Rotating-frame free evolution: H = (Δ/2)·σz  (no drive)."""
    return (delta / 2) * sz


# ============================================================================
# Core Lindblad dissipator and master equation
# ============================================================================

def _dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard Lindblad dissipator: D[L]ρ = LρL† - ½{L†L, ρ}.

    This is the unique Markovian, trace-preserving, completely-positive
    form for a single collapse operator L.

    Parameters
    ----------
    L   : (2,2) Lindblad operator (collapse operator)
    rho : (2,2) density matrix

    Returns
    -------
    (2,2) contribution to dρ/dt from this channel
    """
    Ld   = L.conj().T
    LdL  = Ld @ L
    return L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    noise: NoiseModel,
) -> np.ndarray:
    """Full Lindblad master equation RHS for arbitrary noise channels.

    dρ/dt = -i[H, ρ] + Σ_k γ_k · D[L_k]ρ

    Supports any combination of:
      - Amplitude damping   (L = σ₋,  γ = 1/T₁)
      - Phase damping       (L = σz,  γ = γ_φ)
      - Depolarizing noise  (L ∈ {σx, σy, σz},  γ = γ_dep/4 each)

    Parameters
    ----------
    rho   : (2,2) density matrix
    H     : (2,2) Hamiltonian (ℏ=1 units)
    noise : NoiseModel specifying active channels and rates

    Returns
    -------
    drho_dt : (2,2) complex array — time derivative of ρ
    """
    # Coherent evolution: -i[H, ρ]
    drho = -1j * (H @ rho - rho @ H)

    # Incoherent dissipation: Σ_k γ_k · D[L_k]ρ
    for L, gamma_k in noise.lindblad_operators():
        drho += gamma_k * _dissipator(L, rho)

    return drho


# ============================================================================
# Noise-channel-specific analytic solutions (no Hamiltonian, no drive)
# ============================================================================

def amplitude_damping_analytic(
    rho0: np.ndarray,
    gamma1: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic solution for pure amplitude damping (no drive, no dephasing).

    Exact Kraus-operator solution:
        ρ₀₀(t) = 1 - (1 - ρ₀₀(0)) · e^(-γ₁t)
        ρ₁₁(t) = ρ₁₁(0) · e^(-γ₁t)
        ρ₀₁(t) = ρ₀₁(0) · e^(-γ₁t/2)

    Parameters
    ----------
    rho0   : (2,2) initial density matrix
    gamma1 : amplitude damping rate 1/T₁
    t      : (N,) time array

    Returns
    -------
    t, rx, ry, rz : Bloch vector components at each time
    """
    rho00_0 = float(np.real(rho0[0, 0]))
    rho11_0 = float(np.real(rho0[1, 1]))
    rho01_0 = complex(rho0[0, 1])

    decay   = np.exp(-gamma1 * t)
    rho11_t = rho11_0 * decay
    rho00_t = 1.0 - rho11_t
    rho01_t = rho01_0 * np.exp(-gamma1 * t / 2)

    rx = 2.0 * np.real(rho01_t)
    ry = -2.0 * np.imag(rho01_t)
    rz = rho00_t - rho11_t
    return t, rx, ry, rz


def phase_damping_analytic(
    rho0: np.ndarray,
    gamma_phi: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic solution for pure phase damping (no drive, no amplitude damping).

    Exact solution:
        ρ₀₀(t) = ρ₀₀(0)   [populations constant]
        ρ₁₁(t) = ρ₁₁(0)
        ρ₀₁(t) = ρ₀₁(0) · e^(-2γ_φ·t)

    Note the factor of 2: the D[σz] dissipator contributes 2γ_φ to the
    off-diagonal decay rate (since σz has eigenvalues ±1, each fluctuation
    kicks the phase by ±2).

    Parameters
    ----------
    rho0      : (2,2) initial density matrix
    gamma_phi : pure dephasing rate
    t         : (N,) time array

    Returns
    -------
    t, rx, ry, rz : Bloch vector components
    """
    rho00_0 = float(np.real(rho0[0, 0]))
    rho11_0 = float(np.real(rho0[1, 1]))
    rho01_0 = complex(rho0[0, 1])

    rho01_t = rho01_0 * np.exp(-2.0 * gamma_phi * t)

    rx = 2.0 * np.real(rho01_t)
    ry = -2.0 * np.imag(rho01_t)
    rz = np.full_like(t, rho00_0 - rho11_0)   # rz constant
    return t, rx, ry, rz


def depolarizing_analytic(
    rho0: np.ndarray,
    gamma_dep: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic solution for pure depolarizing channel (no drive, no T1/T2).

    The depolarizing channel shrinks the Bloch vector uniformly:

        r⃗(t) = r⃗(0) · e^(-γ_dep · t)

    This follows because D[σx] + D[σy] + D[σz] applied to the Bloch
    decomposition ρ = (I + r⃗·σ⃗)/2 gives:

        d(r⃗·σ⃗)/dt = -γ_dep · r⃗·σ⃗

    so all three components decay at the same rate, preserving direction.
    The state approaches I/2 (Bloch origin) isotropically.

    Parameters
    ----------
    rho0      : (2,2) initial density matrix
    gamma_dep : depolarizing rate 1/T_dep
    t         : (N,) time array

    Returns
    -------
    t, rx, ry, rz : Bloch vector components
    """
    r0      = dm_to_bloch(rho0)
    decay   = np.exp(-gamma_dep * t)
    rx      = r0[0] * decay
    ry      = r0[1] * decay
    rz      = r0[2] * decay
    return t, rx, ry, rz


# ============================================================================
# ODE solver — general NoiseModel
# ============================================================================

def simulate_dm(
    rho0: np.ndarray,
    H: np.ndarray,
    noise_or_T1,           # NoiseModel  OR  float T1  (legacy positional)
    t_max_or_T2=None,      # float t_max OR  float T2  (legacy positional)
    n_or_tmax=None,        # int   n     OR  float t_max (legacy positional)
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the Lindblad master equation for arbitrary noise channels.

    Accepts two calling conventions so existing code keeps working:

    New API (recommended)
    ---------------------
    simulate_dm(rho0, H, noise_model, t_max, n=1000)

    Legacy API (backward-compatible)
    ---------------------------------
    simulate_dm(rho0, H, T1, T2, t_max, n=1000)
      → internally wraps T1/T2 in NoiseModel(T1=T1, T2=T2)

    Parameters
    ----------
    rho0  : (2,2) initial density matrix
    H     : (2,2) Hamiltonian (time-independent; ℏ=1)
    noise : NoiseModel — selects amplitude damping, phase damping, depolarizing
    t_max : simulation duration
    n     : number of output time points

    Returns
    -------
    t      : (n,)  time axis
    rx     : (n,)  Bloch x component
    ry     : (n,)  Bloch y component
    rz     : (n,)  Bloch z component
    pur    : (n,)  purity Tr(ρ²) — tracks pure→mixed transition
    """
    # --- signature dispatch ---------------------------------------------------
    if isinstance(noise_or_T1, NoiseModel):
        # New API: simulate_dm(rho0, H, noise, t_max, n=...)
        noise = noise_or_T1
        t_max = float(t_max_or_T2)
        if n_or_tmax is not None:
            n = int(n_or_tmax)
        # n already has its default or was passed as keyword
    else:
        # Legacy API: simulate_dm(rho0, H, T1, T2, t_max, n=...)
        T1    = float(noise_or_T1)
        T2    = float(t_max_or_T2)
        t_max = float(n_or_tmax)
        if T2 > T1:
            raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")
        noise = NoiseModel(T1=T1, T2=T2)
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    def _pack(rho):
        return np.concatenate([rho.real.ravel(), rho.imag.ravel()])

    def _unpack(y):
        half = len(y) // 2
        return y[:half].reshape(2, 2) + 1j * y[half:].reshape(2, 2)

    def _rhs(t, y):
        rho  = _unpack(y)
        drho = lindblad_rhs(rho, H, noise)
        return _pack(drho)

    y0     = _pack(rho0)
    t_eval = np.linspace(0, t_max, n)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, t_max),
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Density matrix ODE failed: {sol.message}")

    rx_arr  = np.empty(n)
    ry_arr  = np.empty(n)
    rz_arr  = np.empty(n)
    pur_arr = np.empty(n)

    for i in range(n):
        rho_i      = _unpack(sol.y[:, i])
        r          = dm_to_bloch(rho_i)
        rx_arr[i]  = r[0]
        ry_arr[i]  = r[1]
        rz_arr[i]  = r[2]
        pur_arr[i] = purity(rho_i)

    return sol.t, rx_arr, ry_arr, rz_arr, pur_arr


# ============================================================================
# Backward-compatible wrapper (original API: T1, T2 floats)
# ============================================================================

def simulate_dm_legacy(
    rho0: np.ndarray,
    H: np.ndarray,
    T1: float,
    T2: float,
    t_max: float,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for simulate_dm using T1/T2 directly.

    Equivalent to simulate_dm with NoiseModel(T1=T1, T2=T2).
    Amplitude damping + pure dephasing only (no depolarizing).
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")
    noise = NoiseModel(T1=T1, T2=T2)
    return simulate_dm(rho0, H, noise, t_max, n)


# ============================================================================
# Channel comparison: run all three channels from the same initial state
# ============================================================================

def compare_noise_channels(
    rho0: np.ndarray,
    gamma: float,
    t_max: float,
    n: int = 500,
    use_analytic: bool = True,
) -> dict:
    """Simulate all three noise channels for comparison.

    Runs amplitude damping, phase damping, and depolarizing from the same
    initial state and noise strength, returning Bloch trajectories for each.

    Parameters
    ----------
    rho0         : (2,2) initial density matrix
    gamma        : noise rate (1/time_unit) used for all three channels.
                   Converted to: T1 = 1/γ, T2 = 1/γ, T_dep = 1/γ.
    t_max        : simulation duration
    n            : number of time points
    use_analytic : use closed-form analytic solutions (True, faster)
                   or numerical ODE (False, more general)

    Returns
    -------
    dict with keys:
        't'            : (n,) time axis
        'amplitude'    : dict with 'rx', 'ry', 'rz', 'purity'
        'phase'        : dict with 'rx', 'ry', 'rz', 'purity'
        'depolarizing' : dict with 'rx', 'ry', 'rz', 'purity'
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    t = np.linspace(0, t_max, n)
    T = 1.0 / gamma
    results = {'t': t}

    if use_analytic:
        # Amplitude damping
        _, rx, ry, rz = amplitude_damping_analytic(rho0, gamma, t)
        pur = np.array([purity(bloch_to_dm(np.array([rx[i], ry[i], rz[i]])))
                        for i in range(n)])
        results['amplitude'] = {'rx': rx, 'ry': ry, 'rz': rz, 'purity': pur}

        # Phase damping (pure dephasing: T1 → ∞ so γ_φ = 1/(2T2) and T2 = 1/γ)
        # γ_φ for the analytic function equals γ directly (not the formula-derived γ_φ)
        _, rx, ry, rz = phase_damping_analytic(rho0, gamma / 2.0, t)
        pur = np.array([purity(bloch_to_dm(np.array([rx[i], ry[i], rz[i]])))
                        for i in range(n)])
        results['phase'] = {'rx': rx, 'ry': ry, 'rz': rz, 'purity': pur}

        # Depolarizing
        _, rx, ry, rz = depolarizing_analytic(rho0, gamma, t)
        pur = np.array([purity(bloch_to_dm(np.array([rx[i], ry[i], rz[i]])))
                        for i in range(n)])
        results['depolarizing'] = {'rx': rx, 'ry': ry, 'rz': rz, 'purity': pur}

    else:
        # Numerical ODE path
        H_zero = np.zeros((2, 2), dtype=complex)

        for label, nm in [
            ('amplitude',    noise_amplitude_damping(T)),
            ('phase',        noise_phase_damping(T)),
            ('depolarizing', noise_depolarizing(T)),
        ]:
            _, rx, ry, rz, pur = simulate_dm(rho0, H_zero, nm, t_max, n)
            results[label] = {'rx': rx, 'ry': ry, 'rz': rz, 'purity': pur}

    return results


# ============================================================================
# Purity decay analysis
# ============================================================================

def purity_decay_rates(noise: NoiseModel) -> dict:
    """Compute the rate at which purity decays for each noise channel.

    For a state starting on the equator of the Bloch sphere (|r⃗| = 1):
        - Amplitude damping:  |r_perp| ~ e^(-t/2T1), |rz| ~ 1-e^(-t/T1)
        - Pure dephasing:     |r_perp| ~ e^(-2γ_φ·t), rz unchanged
        - Depolarizing:       |r⃗| ~ e^(-γ_dep·t)

    Purity: Tr(ρ²) = (1 + |r⃗|²) / 2

    Parameters
    ----------
    noise : NoiseModel

    Returns
    -------
    dict with keys:
        'T2_eff'         : effective transverse decay time (all channels)
        'T1_eff'         : effective longitudinal decay time
        'gamma_total'    : total Bloch vector decay rate
        'channels'       : list of active channels
        'per_channel'    : dict of per-channel rates
    """
    g1   = noise._gamma1()
    gphi = noise._gamma_phi()
    gdep = noise._gamma_dep()

    # Transverse: 1/T2_eff = 1/(2T1) + 2γ_φ + γ_dep
    # (factor 2 on γ_φ from D[σz] rate, factor 1 from depolarizing)
    gamma_perp = g1 / 2.0 + 2.0 * gphi + gdep
    # Longitudinal: 1/T1_eff = γ1 + γ_dep
    gamma_long = g1 + gdep

    return {
        'T2_eff':      1.0 / gamma_perp if gamma_perp > 0 else np.inf,
        'T1_eff':      1.0 / gamma_long if gamma_long > 0 else np.inf,
        'gamma_perp':  gamma_perp,
        'gamma_long':  gamma_long,
        'channels':    noise.channels,
        'per_channel': {
            'amplitude_damping': {'gamma': g1,   'T_eff': 1.0 / g1   if g1   > 0 else np.inf},
            'phase_damping':     {'gamma': gphi,  'T_eff': 1.0 / gphi if gphi > 0 else np.inf},
            'depolarizing':      {'gamma': gdep,  'T_eff': 1.0 / gdep if gdep > 0 else np.inf},
        },
    }


# ============================================================================
# Cross-validation: density matrix vs Bloch ODE (amplitude+phase damping only)
# ============================================================================

def bloch_vs_dm_error(
    omega_rabi: float,
    delta: float,
    T1: float,
    T2: float,
    t_max: float,
    n: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compare Bloch ODE with Lindblad density matrix (amplitude + phase damping).

    Parameters
    ----------
    omega_rabi : Rabi drive strength Ω
    delta      : detuning Δ
    T1, T2     : relaxation times
    t_max      : simulation duration
    n          : number of time points

    Returns
    -------
    t         : time axis
    bloch_dm  : (n, 3) Bloch vectors from density matrix
    max_error : max |r_bloch - r_dm| across all times and components
    """
    from .rabi import run_rabi

    H    = hamiltonian_rotating(delta, omega_rabi)
    rho0 = ground_state_dm()
    noise = NoiseModel(T1=T1, T2=T2)

    t_dm, rx, ry, rz, _ = simulate_dm(rho0, H, noise, t_max, n=n)
    bloch_dm = np.column_stack([rx, ry, rz])

    t_bl, Mx, My, Mz = run_rabi(omega_rabi, delta, T1, T2, t_max, n=n)
    bloch_ode = np.column_stack([Mx, My, Mz])

    max_error = float(np.max(np.abs(bloch_dm - bloch_ode)))
    return t_dm, bloch_dm, max_error


# ============================================================================
# lindblad_rhs — legacy alias (T1, T2 float API)
# ============================================================================

def lindblad_rhs_legacy(
    rho: np.ndarray,
    H: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Legacy T1/T2-based Lindblad RHS (amplitude damping + pure dephasing).

    Equivalent to lindblad_rhs(rho, H, NoiseModel(T1=T1, T2=T2)).
    Kept for backward compatibility.
    """
    noise = NoiseModel(T1=T1, T2=T2)
    return lindblad_rhs(rho, H, noise)