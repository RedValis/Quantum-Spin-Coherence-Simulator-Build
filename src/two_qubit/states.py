"""
states.py — Two-qubit state space: basis, Bell states, tensor products.
=======================================================================

Two-qubit Hilbert space H = H_A ⊗ H_B  is 4-dimensional.
The computational basis ordered as |00⟩, |01⟩, |10⟩, |11⟩ (big-endian:
qubit A is the most-significant bit).

Basis vector index map
----------------------
    |00⟩  →  index 0   A=↑, B=↑
    |01⟩  →  index 1   A=↑, B=↓
    |10⟩  →  index 2   A=↓, B=↑
    |11⟩  →  index 3   A=↓, B=↓

Bell states (maximally entangled)
----------------------------------
    |Φ+⟩ = (|00⟩ + |11⟩)/√2   — zero total spin, symmetric
    |Φ-⟩ = (|00⟩ - |11⟩)/√2   — zero total spin, antisymmetric
    |Ψ+⟩ = (|01⟩ + |10⟩)/√2   — triplet S=1, Sz=0
    |Ψ-⟩ = (|01⟩ - |10⟩)/√2   — singlet S=0 (antisymmetric)

The Bell states form an orthonormal basis for H and are all maximally
entangled: tracing out either qubit gives I/2 (maximally mixed).

Tensor product notation
-----------------------
For states |ψ_A⟩ ∈ H_A and |ψ_B⟩ ∈ H_B:
    |ψ_A⟩ ⊗ |ψ_B⟩  →  kron(psi_A, psi_B)  in NumPy

For operators O_A on H_A only:
    O_A ⊗ I_B  →  kron(O_A, I2)
    I_A ⊗ O_B  →  kron(I2, O_B)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict

# ============================================================================
# Single-qubit basis (column vectors)
# ============================================================================

KET_0 = np.array([1.0, 0.0], dtype=complex)   # |0⟩ = |↑⟩ ground
KET_1 = np.array([0.0, 1.0], dtype=complex)   # |1⟩ = |↓⟩ excited

# ============================================================================
# Two-qubit computational basis  (4-component column vectors)
# ============================================================================

KET_00 = np.kron(KET_0, KET_0)   # [1, 0, 0, 0]
KET_01 = np.kron(KET_0, KET_1)   # [0, 1, 0, 0]
KET_10 = np.kron(KET_1, KET_0)   # [0, 0, 1, 0]
KET_11 = np.kron(KET_1, KET_1)   # [0, 0, 0, 1]

BASIS_LABELS = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
BASIS_KETS   = [KET_00, KET_01, KET_10, KET_11]

# ============================================================================
# Single-qubit Pauli matrices (re-exported for convenience)
# ============================================================================

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

sx = np.array([[0, 1], [1,  0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

# ============================================================================
# Tensor product helpers
# ============================================================================

def tensor(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tensor (Kronecker) product A ⊗ B.

    Works for both state vectors (shape (m,) ⊗ (n,) → (mn,))
    and operators (shape (m,m) ⊗ (n,n) → (mn,mn)).

    Parameters
    ----------
    A, B : arrays of compatible rank (both 1-D or both 2-D)

    Returns
    -------
    np.ndarray   A ⊗ B
    """
    return np.kron(A, B)


def tensor_op_on_qubit(O: np.ndarray, qubit: int, n_qubits: int = 2) -> np.ndarray:
    """Embed single-qubit operator O acting on `qubit` into n_qubit space.

    For n_qubits=2:
        qubit=0 → O ⊗ I
        qubit=1 → I ⊗ O

    Parameters
    ----------
    O        : (2,2) single-qubit operator
    qubit    : index of the target qubit (0 = most significant)
    n_qubits : total number of qubits (default 2)

    Returns
    -------
    (2^n, 2^n) operator in full Hilbert space
    """
    ops = [I2] * n_qubits
    ops[qubit] = O
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def ket_to_dm(psi: np.ndarray) -> np.ndarray:
    """Convert state vector |ψ⟩ to density matrix ρ = |ψ⟩⟨ψ|.

    Parameters
    ----------
    psi : (N,) complex state vector (need not be pre-normalised)

    Returns
    -------
    rho : (N,N) density matrix, normalised so Tr(ρ)=1
    """
    psi = np.asarray(psi, dtype=complex)
    norm_sq = np.real(psi.conj() @ psi)
    if norm_sq < 1e-14:
        raise ValueError("State vector has zero norm")
    psi = psi / np.sqrt(norm_sq)
    return np.outer(psi, psi.conj())


# ============================================================================
# Bell states
# ============================================================================

def bell_state(which: str = "phi_plus") -> np.ndarray:
    """Return a Bell state as a (4,) complex state vector.

    The four Bell states form a maximally entangled orthonormal basis
    for the two-qubit Hilbert space.  Each has concurrence C=1 and
    entanglement entropy E=1 ebit.

    Parameters
    ----------
    which : str — one of:
        'phi_plus'  |Φ+⟩ = (|00⟩ + |11⟩)/√2   (default)
        'phi_minus' |Φ-⟩ = (|00⟩ - |11⟩)/√2
        'psi_plus'  |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        'psi_minus' |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    Returns
    -------
    psi : (4,) complex array,  normalised
    """
    _bells = {
        'phi_plus':  (KET_00 + KET_11) / np.sqrt(2),
        'phi_minus': (KET_00 - KET_11) / np.sqrt(2),
        'psi_plus':  (KET_01 + KET_10) / np.sqrt(2),
        'psi_minus': (KET_01 - KET_10) / np.sqrt(2),
    }
    if which not in _bells:
        raise ValueError(
            f"Bell state '{which}' not recognised. "
            f"Choose from: {list(_bells)}"
        )
    return _bells[which].copy()


def all_bell_states() -> Dict[str, np.ndarray]:
    """Return dict of all four Bell state vectors."""
    return {k: bell_state(k) for k in ('phi_plus', 'phi_minus',
                                        'psi_plus', 'psi_minus')}


def bell_state_dm(which: str = "phi_plus") -> np.ndarray:
    """Return a Bell state as a (4,4) density matrix."""
    return ket_to_dm(bell_state(which))


# ============================================================================
# Product state constructors
# ============================================================================

def product_state(
    theta_A: float, phi_A: float,
    theta_B: float, phi_B: float,
) -> np.ndarray:
    """Two-qubit product (separable) state |ψ_A⟩ ⊗ |ψ_B⟩.

    Each qubit is a pure state on the Bloch sphere:
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

    Parameters
    ----------
    theta_A, phi_A : Bloch angles for qubit A
    theta_B, phi_B : Bloch angles for qubit B

    Returns
    -------
    psi : (4,) normalised state vector
    """
    def _bloch_ket(theta, phi):
        return np.array([np.cos(theta/2),
                         np.exp(1j*phi) * np.sin(theta/2)], dtype=complex)
    return np.kron(_bloch_ket(theta_A, phi_A), _bloch_ket(theta_B, phi_B))


def computational_basis_state(n: int) -> np.ndarray:
    """Return the n-th computational basis state as a (4,) vector.

    n=0 → |00⟩,  n=1 → |01⟩,  n=2 → |10⟩,  n=3 → |11⟩
    """
    if n not in range(4):
        raise ValueError(f"n must be 0–3, got {n}")
    psi = np.zeros(4, dtype=complex)
    psi[n] = 1.0
    return psi


def random_pure_state(seed: int | None = None) -> np.ndarray:
    """Haar-random two-qubit pure state.

    Samples uniformly from the unit sphere in ℂ⁴ by drawing a complex
    Gaussian vector and normalising.

    Parameters
    ----------
    seed : optional RNG seed for reproducibility

    Returns
    -------
    psi : (4,) complex unit vector
    """
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    return psi / np.linalg.norm(psi)


def ghz_like_state(n_qubits: int = 2) -> np.ndarray:
    """GHZ-like state (|00...0⟩ + |11...1⟩)/√2 for n qubits.

    For n_qubits=2 this is exactly |Φ+⟩.

    Parameters
    ----------
    n_qubits : number of qubits (≥ 2)

    Returns
    -------
    psi : (2^n,) complex array
    """
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=complex)
    psi[0]    = 1.0 / np.sqrt(2)   # |00...0⟩
    psi[-1]   = 1.0 / np.sqrt(2)   # |11...1⟩
    return psi


# ============================================================================
# State properties
# ============================================================================

def inner_product(psi: np.ndarray, phi: np.ndarray) -> complex:
    """⟨ψ|φ⟩ — complex inner product."""
    return complex(np.conj(psi) @ phi)


def fidelity_pure(psi: np.ndarray, phi: np.ndarray) -> float:
    """Fidelity F = |⟨ψ|φ⟩|² between two pure states."""
    return float(np.abs(inner_product(psi, phi))**2)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(ρ,σ) = (Tr√(√ρ σ √ρ))² between mixed states.

    For pure states ρ=|ψ⟩⟨ψ|, σ=|φ⟩⟨φ| this reduces to |⟨ψ|φ⟩|².

    Parameters
    ----------
    rho, sigma : (N,N) density matrices

    Returns
    -------
    float in [0, 1]
    """
    # √ρ via eigendecomposition
    evals, evecs = np.linalg.eigh(rho)
    evals = np.maximum(evals, 0.0)   # numerical safety
    sqrt_rho = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    M    = sqrt_rho @ sigma @ sqrt_rho
    evals_M = np.linalg.eigvalsh(M)
    evals_M = np.maximum(evals_M, 0.0)
    return float(np.sum(np.sqrt(evals_M))**2)


def is_entangled(psi: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a two-qubit pure state is entangled.

    A pure state is separable iff its Schmidt rank is 1, equivalently
    iff the reduced density matrix has rank 1 (pure reduced state).
    We test whether the smaller Schmidt coefficient is above *tol*.

    Parameters
    ----------
    psi : (4,) state vector
    tol : threshold for Schmidt coefficient significance

    Returns
    -------
    bool — True if entangled
    """
    # Schmidt decomposition via SVD of the coefficient matrix
    M    = psi.reshape(2, 2)
    svd  = np.linalg.svd(M, compute_uv=False)   # singular values (≥0, sorted desc)
    return bool(svd[1] > tol)


def state_summary(psi: np.ndarray) -> dict:
    """Return a dict of key properties of a two-qubit pure state.

    Parameters
    ----------
    psi : (4,) state vector

    Returns
    -------
    dict with keys:
        'amplitudes'    : dict {label: complex amplitude}
        'probabilities' : dict {label: |amplitude|²}
        'is_entangled'  : bool
        'bell_fidelities': dict {bell_label: fidelity with each Bell state}
    """
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    rho = ket_to_dm(psi)

    amps  = {lbl: complex(psi[i]) for i, lbl in enumerate(BASIS_LABELS)}
    probs = {lbl: float(abs(v)**2)  for lbl, v in amps.items()}

    bell_fids = {}
    for bname, bket in all_bell_states().items():
        bell_fids[bname] = fidelity_pure(psi, bket)

    return {
        'amplitudes':      amps,
        'probabilities':   probs,
        'is_entangled':    is_entangled(psi),
        'bell_fidelities': bell_fids,
    }
