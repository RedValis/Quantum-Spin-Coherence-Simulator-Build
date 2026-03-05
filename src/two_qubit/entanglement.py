"""
entanglement.py - Entanglement measures for two-qubit systems.
==============================================================

A bipartite state ρ_AB is *entangled* if it cannot be written as a
convex mixture of product states:  ρ ≠ Σ_i p_i ρ_A^i ⊗ ρ_B^i.

Measures implemented
---------------------

1. Partial trace  Tr_B[ρ_AB] → ρ_A  (reduced density matrix)
   The partial trace is the quantum analogue of marginalising a joint
   probability distribution.  If ρ_AB is entangled, ρ_A is mixed.

2. Von Neumann entropy  S(ρ) = -Tr(ρ log₂ ρ)
   S=0 for pure states, S=1 for the maximally mixed qubit state I/2.
   For a bipartite pure state |ψ_AB⟩, the entanglement entropy is
   E = S(ρ_A) = S(ρ_B) (always equal for pure states).

3. Entanglement entropy  E(|ψ⟩) = S(Tr_B[|ψ⟩⟨ψ|])
   Ranges from 0 (product state) to 1 ebit (maximally entangled Bell state).
   Only defined for pure states; use concurrence for mixed states.

4. Concurrence  C(ρ) - Wootters 1998
   The concurrence is defined for *mixed* two-qubit states and provides
   a closed-form entanglement of formation formula.  For pure states:
       C(|ψ⟩) = |⟨ψ|ψ̃⟩|  where |ψ̃⟩ = (σy⊗σy)|ψ*⟩
   For mixed states the formula uses the eigenvalues of R = ρ(σy⊗σy)ρ*(σy⊗σy).
   C=0 → separable, C=1 → maximally entangled.

5. Schmidt decomposition  |ψ_AB⟩ = Σ_k λ_k |a_k⟩⊗|b_k⟩
   Schmidt rank=1 ↔ separable pure state.
   Schmidt coefficients λ_k (non-negative, sum of squares = 1).

6. Negativity  N(ρ) - Vidal & Werner 2002
   N(ρ) = (‖ρ^{T_B}‖₁ - 1) / 2
   where ρ^{T_B} is the partial transpose.  N>0 is sufficient (but not
   always necessary) for entanglement (PPT criterion).

7. Entanglement of Formation  E_F(C) = h((1+√(1-C²))/2)
   where h is the binary entropy.  Exact for two-qubit states.

8. Bell inequality (CHSH) witness
   B = ⟨σz⊗σz⟩ + ⟨σz⊗σx⟩ + ⟨σx⊗σz⟩ - ⟨σx⊗σx⟩  ≤ 2 classically
   Quantum maximum: 2√2 (Tsirelson's bound).  B > 2 → entangled.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict

from .states import I2, I4, sx, sy, sz, tensor, ket_to_dm, KET_00, KET_11


# ============================================================================
# Partial trace
# ============================================================================

def partial_trace(rho: np.ndarray, keep: int) -> np.ndarray:
    """Partial trace over the *other* qubit, keeping qubit *keep*.

    For a 4×4 two-qubit density matrix in the basis |00⟩,|01⟩,|10⟩,|11⟩:

        keep=0 → trace out qubit B → return ρ_A = Tr_B[ρ]  (2×2)
        keep=1 → trace out qubit A → return ρ_B = Tr_A[ρ]  (2×2)

    The partial trace is computed by reshaping ρ into a (2,2,2,2) tensor
    ρ[a,b,a',b'] = ⟨a,b|ρ|a',b'⟩  and summing over the traced qubit index.

    Parameters
    ----------
    rho  : (4,4) two-qubit density matrix
    keep : 0 to keep qubit A, 1 to keep qubit B

    Returns
    -------
    rho_reduced : (2,2) reduced density matrix
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"Expected (4,4) density matrix, got {rho.shape}")

    # Reshape into (d_A, d_B, d_A, d_B) = (2, 2, 2, 2)
    R = rho.reshape(2, 2, 2, 2)   # R[a, b, a', b']

    if keep == 0:
        # ρ_A[a, a'] = Σ_b  R[a, b, a', b]
        return np.einsum('abac->bc', R.transpose(0, 1, 2, 3)) if False else \
               np.einsum('abab->aa', R.reshape(2, 2, 2, 2).transpose(0, 1, 2, 3)) if False else \
               np.trace(R.transpose(0, 2, 1, 3).reshape(4, 4)).reshape(1) if False else \
               np.einsum('iajb->ij', R.reshape(2, 2, 2, 2))   # sum over j=b
    elif keep == 1:
        # ρ_B[b, b'] = Σ_a  R[a, b, a, b']
        return np.einsum('aiba->ib', R.reshape(2, 2, 2, 2)).T if False else \
               np.einsum('aibj->ij', R.reshape(2, 2, 2, 2))   # sum over i=a
    else:
        raise ValueError(f"keep must be 0 or 1, got {keep}")


def partial_trace_clean(rho: np.ndarray, keep: int) -> np.ndarray:
    """Clean partial trace using explicit index contraction.

    Avoids the reshape gymnastics with a direct loop - clearer physics.
    Identical output to partial_trace() but easier to verify.
    """
    rho = np.asarray(rho, dtype=complex)
    d   = 2   # qubit dimension
    rho4 = rho.reshape(d, d, d, d)   # rho4[a, b, a', b']

    if keep == 0:
        # ρ_A[a, a'] = Σ_b ρ[a,b,a',b]
        rho_A = np.zeros((d, d), dtype=complex)
        for b in range(d):
            rho_A += rho4[:, b, :, b]
        return rho_A
    else:
        # ρ_B[b, b'] = Σ_a ρ[a,b,a,b']
        rho_B = np.zeros((d, d), dtype=complex)
        for a in range(d):
            rho_B += rho4[a, :, a, :]
        return rho_B


# override with clean version for clarity
partial_trace = partial_trace_clean


# ============================================================================
# Von Neumann entropy
# ============================================================================

def von_neumann_entropy(rho: np.ndarray, base: int = 2) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log_b ρ).

    Computes the entropy via eigendecomposition:
        S = -Σ_i λ_i log_b(λ_i)    (0·log(0) ≡ 0 by L'Hôpital)

    Parameters
    ----------
    rho  : (N,N) density matrix
    base : logarithm base (2 for bits/ebits, e for nats)

    Returns
    -------
    float  S ∈ [0, log_b(N)]
        S=0   pure state
        S=1   maximally mixed qubit state (base=2)
    """
    evals = np.linalg.eigvalsh(rho)
    evals = np.maximum(evals, 0.0)          # numerical safety
    evals = evals[evals > 1e-15]            # drop zeros (0 log 0 = 0)
    log_fn = np.log2 if base == 2 else np.log
    return float(-np.sum(evals * log_fn(evals)))


# ============================================================================
# Schmidt decomposition
# ============================================================================

def schmidt_decomposition(
    psi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Schmidt decomposition of a two-qubit pure state.

    Any bipartite pure state |ψ_AB⟩ ∈ H_A ⊗ H_B can be written as:

        |ψ⟩ = Σ_k  λ_k  |α_k⟩ ⊗ |β_k⟩

    where λ_k ≥ 0 are the Schmidt coefficients (sorted descending),
    Σ λ_k² = 1, and {|α_k⟩}, {|β_k⟩} are orthonormal bases on H_A, H_B.

    The coefficients λ_k are the singular values of the (2×2) coefficient
    matrix C[i,j] = ψ[2i+j].

    Key properties:
        Schmidt rank 1   ↔  separable  (λ₁=1, λ₂=0)
        Schmidt rank 2   ↔  entangled
        λ₁=λ₂=1/√2     ↔  maximally entangled (Bell state)
        Entanglement entropy E = -Σ λ_k² log₂(λ_k²)

    Parameters
    ----------
    psi : (4,) normalised state vector

    Returns
    -------
    lambdas : (2,) Schmidt coefficients (non-negative, sorted descending)
    A_vecs  : (2,2) rows are Schmidt basis vectors for qubit A
    B_vecs  : (2,2) rows are Schmidt basis vectors for qubit B
    """
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)

    # Coefficient matrix: C[i,j] = ψ[2i+j]
    C = psi.reshape(2, 2)

    # SVD: C = U @ diag(s) @ Vh
    U, s, Vh = np.linalg.svd(C)

    return s, U.T, Vh   # (lambdas, A_vecs, B_vecs)


def schmidt_number(psi: np.ndarray, tol: float = 1e-8) -> int:
    """Schmidt rank of a two-qubit pure state (1=separable, 2=entangled)."""
    lambdas, _, _ = schmidt_decomposition(psi)
    return int(np.sum(lambdas > tol))


def entanglement_entropy(psi: np.ndarray) -> float:
    """Entanglement entropy E(|ψ⟩) = S(ρ_A) = -Σ λ_k² log₂(λ_k²).

    Ranges from 0 (product state) to 1 ebit (maximally entangled Bell state).

    Relation to Schmidt decomposition:
        E = von_neumann_entropy(Tr_B[|ψ⟩⟨ψ|])
          = -Σ_k  λ_k²  log₂(λ_k²)

    Parameters
    ----------
    psi : (4,) state vector

    Returns
    -------
    float in [0, 1]
    """
    lambdas, _, _ = schmidt_decomposition(psi)
    probs = lambdas**2
    probs = probs[probs > 1e-15]
    return float(-np.sum(probs * np.log2(probs)))


# ============================================================================
# Concurrence (Wootters 1998) - works for mixed states
# ============================================================================

def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence C(ρ) for an arbitrary two-qubit density matrix.

    Definition
    ----------
    The "spin-flipped" state is  ρ̃ = (σy⊗σy) ρ* (σy⊗σy)  where * is
    complex conjugation in the computational basis.

    The concurrence is:
        C(ρ) = max(0, λ₁ - λ₂ - λ₃ - λ₄)

    where λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄ ≥ 0 are the square roots of the eigenvalues
    of R = ρ · ρ̃  (not necessarily Hermitian, but eigenvalues are real).

    For a pure state |ψ⟩:
        C = |⟨ψ|(σy⊗σy)|ψ*⟩|
    which for |Φ+⟩ = (|00⟩+|11⟩)/√2 gives C=1.

    Values
    ------
    C = 0 : separable (not necessarily for mixed states - necessary but
            not sufficient condition via Peres-Horodecki)
    C = 1 : maximally entangled
    C ∈ (0,1) : partially entangled

    Parameters
    ----------
    rho : (4,4) density matrix (Hermitian, trace-1, positive semi-definite)

    Returns
    -------
    float in [0, 1]
    """
    rho = np.asarray(rho, dtype=complex)

    # σy⊗σy in the computational basis
    sysy = np.kron(sy, sy)

    # Spin-flipped density matrix
    rho_tilde = sysy @ rho.conj() @ sysy

    # R = ρ · ρ̃  - eigenvalues are real and non-negative
    R = rho @ rho_tilde

    # Eigenvalues of R (may be complex due to numerics; take real part)
    evals = np.linalg.eigvals(R)
    evals = np.real(evals)
    evals = np.maximum(evals, 0.0)
    lambdas = np.sort(np.sqrt(evals))[::-1]   # sorted descending

    C = float(max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))
    return C


# ============================================================================
# Entanglement of Formation
# ============================================================================

def _binary_entropy(p: float) -> float:
    """Binary entropy h(p) = -p·log₂p - (1-p)·log₂(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def entanglement_of_formation(rho: np.ndarray) -> float:
    """Entanglement of Formation E_F(ρ) for a two-qubit mixed state.

    Uses the Wootters formula (exact for two qubits):

        E_F = h((1 + √(1-C²)) / 2)

    where C = concurrence(ρ) and h is the binary entropy.

    E_F = 0 → separable, E_F = 1 → maximally entangled.

    Parameters
    ----------
    rho : (4,4) density matrix

    Returns
    -------
    float in [0, 1]
    """
    C = concurrence(rho)
    p = (1.0 + np.sqrt(max(0.0, 1.0 - C**2))) / 2.0
    return _binary_entropy(p)


# ============================================================================
# Negativity (PPT criterion)
# ============================================================================

def partial_transpose(rho: np.ndarray, qubit: int = 1) -> np.ndarray:
    """Partial transpose of a two-qubit density matrix.

    Transposes qubit *qubit* while leaving the other qubit unchanged.
    For a state in basis |ab⟩:
        (ρ^{T_B})[a,b,a',b'] = ρ[a,b',a',b]   (B indices swapped)

    The Peres-Horodecki (PPT) criterion states: if ρ^{T_B} has any
    negative eigenvalue, then ρ is entangled.  For 2×2 systems the
    converse also holds (PPT iff separable).

    Parameters
    ----------
    rho   : (4,4) density matrix
    qubit : which qubit to transpose (0=A, 1=B)

    Returns
    -------
    (4,4) partially transposed matrix (not necessarily positive)
    """
    rho = np.asarray(rho, dtype=complex)
    R   = rho.reshape(2, 2, 2, 2)   # R[a, b, a', b']

    if qubit == 1:
        # Transpose B: swap b ↔ b'
        PT = R.transpose(0, 3, 2, 1)
    else:
        # Transpose A: swap a ↔ a'
        PT = R.transpose(2, 1, 0, 3)

    return PT.reshape(4, 4)


def negativity(rho: np.ndarray) -> float:
    """Negativity N(ρ) = (‖ρ^{T_B}‖₁ - 1) / 2.

    The trace norm ‖M‖₁ = Tr(√(M†M)) = sum of singular values.
    For a Hermitian matrix this equals the sum of absolute eigenvalues.

    N = 0 → PPT state (separable for 2-qubit systems)
    N > 0 → entangled (sum of the negative eigenvalues of ρ^{T_B})

    Equivalently: N = -Σ_{λ<0} λ  (sum of negative eigenvalues of ρ^{T_B})

    Parameters
    ----------
    rho : (4,4) density matrix

    Returns
    -------
    float ≥ 0
    """
    PT     = partial_transpose(rho, qubit=1)
    evals  = np.linalg.eigvalsh(PT)
    return float(np.sum(np.abs(evals[evals < 0])))


def logarithmic_negativity(rho: np.ndarray) -> float:
    """Logarithmic negativity E_N = log₂(‖ρ^{T_B}‖₁).

    An entanglement monotone that upper-bounds distillable entanglement.
    E_N = 0 for separable states; E_N = 1 for maximally entangled 2-qubit states.

    Parameters
    ----------
    rho : (4,4) density matrix

    Returns
    -------
    float ≥ 0
    """
    PT       = partial_transpose(rho, qubit=1)
    evals    = np.linalg.eigvalsh(PT)
    trace_norm = np.sum(np.abs(evals))
    return float(np.log2(max(trace_norm, 1e-15)))


# ============================================================================
# CHSH Bell inequality witness
# ============================================================================

def chsh_value(rho: np.ndarray) -> float:
    """CHSH value for rho using Tsirelson-optimal measurement settings.

    Uses: A1=sz, A2=sx, B1=(sz+sx)/sqrt(2), B2=(sz-sx)/sqrt(2).
    B(|Phi+>) = 2*sqrt(2) ~ 2.828 (Tsirelson bound).
    Classical bound: B <= 2.  B > 2 implies entangled.

    Parameters
    ----------
    rho : (4,4) density matrix

    Returns
    -------
    float  B value  (> 2 implies entangled)
    """
    def _expect(O):
        return float(np.real(np.trace(O @ rho)))

    A1 = sz
    A2 = sx
    B1 = (sz + sx) / np.sqrt(2)
    B2 = (sz - sx) / np.sqrt(2)
    chsh_op = (np.kron(A1, B1) + np.kron(A1, B2)
             + np.kron(A2, B1) - np.kron(A2, B2))
    return _expect(chsh_op)


# ============================================================================
# Combined summary
# ============================================================================

def entanglement_summary(
    state: np.ndarray,
) -> Dict[str, float]:
    """Compute all entanglement measures for a two-qubit state.

    Accepts either a (4,) state vector or (4,4) density matrix.

    Parameters
    ----------
    state : (4,) pure state vector  OR  (4,4) density matrix

    Returns
    -------
    dict with keys:
        'concurrence'          : Wootters C ∈ [0, 1]
        'entanglement_entropy' : E(ψ) ∈ [0, 1]  (pure states only)
        'entanglement_of_formation' : E_F ∈ [0, 1]
        'negativity'           : N ≥ 0
        'log_negativity'       : E_N ≥ 0
        'chsh_value'           : B (> 2 implies entangled)
        'purity_A'             : Tr(ρ_A²)  ← 0.5 = max entangled
        'purity_B'             : Tr(ρ_B²)
        'entropy_A'            : S(ρ_A)
        'entropy_B'            : S(ρ_B)
        'schmidt_number'       : rank  (pure states only)
        'is_entangled'         : bool (concurrence > 1e-6)
    """
    state = np.asarray(state, dtype=complex)

    if state.ndim == 1:
        # Pure state: state vector
        psi = state / np.linalg.norm(state)
        rho = ket_to_dm(psi)
        is_pure = True
    else:
        rho = state
        # Check if pure
        pur = float(np.real(np.trace(rho @ rho)))
        is_pure = (pur > 1.0 - 1e-6)
        if is_pure:
            evals, evecs = np.linalg.eigh(rho)
            psi = evecs[:, np.argmax(evals)]

    rho_A = partial_trace(rho, keep=0)
    rho_B = partial_trace(rho, keep=1)

    C    = concurrence(rho)
    E_of = entanglement_of_formation(rho)
    neg  = negativity(rho)
    lneg = logarithmic_negativity(rho)
    chsh = chsh_value(rho)
    purA = float(np.real(np.trace(rho_A @ rho_A)))
    purB = float(np.real(np.trace(rho_B @ rho_B)))
    SA   = von_neumann_entropy(rho_A)
    SB   = von_neumann_entropy(rho_B)

    result = {
        'concurrence':               C,
        'entanglement_of_formation': E_of,
        'negativity':                neg,
        'log_negativity':            lneg,
        'chsh_value':                chsh,
        'purity_A':                  purA,
        'purity_B':                  purB,
        'entropy_A':                 SA,
        'entropy_B':                 SB,
        'is_entangled':              C > 1e-6,
    }

    if is_pure:
        result['entanglement_entropy'] = entanglement_entropy(psi)
        result['schmidt_number']       = schmidt_number(psi)

    return result


# ============================================================================
# Entanglement dynamics helper
# ============================================================================

def track_entanglement(
    rho_t: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute entanglement measures over a trajectory of density matrices.

    Parameters
    ----------
    rho_t : (T, 4, 4) array of density matrices at T time points

    Returns
    -------
    dict with time-series arrays:
        'concurrence'           : (T,)
        'negativity'            : (T,)
        'entropy_A'             : (T,)
        'entanglement_of_formation' : (T,)
        'purity_A'              : (T,)
        'purity_B'              : (T,)
    """
    T     = rho_t.shape[0]
    out   = {k: np.empty(T) for k in (
        'concurrence', 'negativity', 'entropy_A',
        'entanglement_of_formation', 'purity_A', 'purity_B',
    )}

    for i in range(T):
        rho   = rho_t[i]
        rho_A = partial_trace(rho, keep=0)
        rho_B = partial_trace(rho, keep=1)

        out['concurrence'][i]               = concurrence(rho)
        out['negativity'][i]                = negativity(rho)
        out['entropy_A'][i]                 = von_neumann_entropy(rho_A)
        out['entanglement_of_formation'][i] = entanglement_of_formation(rho)
        out['purity_A'][i]                  = float(np.real(np.trace(rho_A @ rho_A)))
        out['purity_B'][i]                  = float(np.real(np.trace(rho_B @ rho_B)))

    return out
