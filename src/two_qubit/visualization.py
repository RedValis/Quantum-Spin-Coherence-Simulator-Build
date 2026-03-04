"""
visualization.py — Two-qubit entanglement and density matrix visualizations.
=============================================================================

Plot functions
--------------
plot_dm_heatmap        : Real/imag/abs density matrix as annotated heatmap
plot_hinton            : Hinton diagram (squares scaled by element magnitude)
plot_entanglement_evolution : Concurrence + entropy vs time
plot_bloch_pair        : Side-by-side Bloch spheres for the two reduced states
plot_bell_fidelities   : Bar chart of fidelity with each Bell state
plot_concurrence_phase : Concurrence as a function of a circuit angle (sweep)
plot_esd_demo          : Entanglement sudden death figure
plot_dm_trajectory     : Animated or multi-panel DM snapshots over time
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Sequence, Dict, Tuple


# ============================================================================
# Helpers
# ============================================================================

BASIS_LABELS = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

_CMAP_POS  = cm.get_cmap("Blues")
_CMAP_DIV  = cm.get_cmap("RdBu_r")
_CMAP_ABS  = cm.get_cmap("viridis")

PALETTE = {
    'concurrence':  '#E63946',
    'entropy':      '#2C7BB6',
    'negativity':   '#6A994E',
    'purity':       '#FF9F1C',
    'fidelity':     '#7B2D8B',
}


def _annotate_heatmap(ax, data, fmt=".2f", threshold=None, fontsize=8):
    """Add text annotations to a heatmap."""
    n = data.shape[0]
    if threshold is None:
        threshold = data.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            color = "white" if abs(val) > threshold else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=fontsize, color=color)


# ============================================================================
# Density matrix heatmap
# ============================================================================

def plot_dm_heatmap(
    rho: np.ndarray,
    title: str = "Density Matrix",
    show_imag: bool = True,
    show_abs: bool = False,
    save_path: Optional[str] = None,
    annotate: bool = True,
    fontsize: int = 9,
) -> plt.Figure:
    """Visualise a two-qubit density matrix as annotated heatmaps.

    Shows up to three panels: Re(ρ), Im(ρ), and |ρ|.
    The diagonal (populations) and off-diagonals (coherences) are
    visually distinguished by their position.

    Parameters
    ----------
    rho        : (4,4) density matrix
    title      : figure title
    show_imag  : include the imaginary part panel (default True)
    show_abs   : include the absolute value panel (default False)
    save_path  : save figure to this path if given
    annotate   : overlay numerical values on each cell
    fontsize   : annotation font size

    Returns
    -------
    matplotlib.figure.Figure
    """
    rho = np.asarray(rho, dtype=complex)
    n_panels = 1 + int(show_imag) + int(show_abs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]

    data_parts = [("Re(ρ)", np.real(rho), _CMAP_DIV, (-1, 1))]
    if show_imag:
        data_parts.append(("Im(ρ)", np.imag(rho), _CMAP_DIV, (-1, 1)))
    if show_abs:
        data_parts.append(("|ρ|", np.abs(rho), _CMAP_ABS, (0, 1)))

    for ax, (label, data, cmap, (vmin, vmax)) in zip(axes, data_parts):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(BASIS_LABELS, fontsize=8, rotation=45, ha='right')
        ax.set_yticklabels(BASIS_LABELS, fontsize=8)
        ax.set_title(label, fontsize=10, fontweight='bold', pad=8)

        # Diagonal box
        for k in range(4):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                         fill=False, edgecolor='#333333', lw=1.2, zorder=3))

        if annotate:
            _annotate_heatmap(ax, data, fmt=".3f", fontsize=fontsize)

        ax.set_facecolor('#F5F5F5')

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Hinton diagram
# ============================================================================

def plot_hinton(
    rho: np.ndarray,
    title: str = "Hinton Diagram — Density Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Hinton diagram: squares whose area encodes |ρ_ij|, colour encodes phase.

    White squares = positive real; black squares = negative real.
    Coloured squares = complex elements (hue = phase angle).

    The Hinton diagram was originally used for weight matrices in neural
    networks but is a natural way to visualise density matrices where
    both magnitude and phase carry physical meaning.

    Parameters
    ----------
    rho       : (4,4) density matrix
    title     : plot title
    save_path : save PNG if given

    Returns
    -------
    matplotlib.figure.Figure
    """
    rho  = np.asarray(rho, dtype=complex)
    n    = rho.shape[0]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    max_val = np.max(np.abs(rho))
    scale   = 0.9 / max_val if max_val > 0 else 1.0

    for i in range(n):
        for j in range(n):
            val  = rho[i, j]
            size = np.sqrt(np.abs(val)) * scale   # area ∝ |ρ|, so side ∝ √|ρ|
            phase = np.angle(val)

            # Map phase to colour (HSL wheel)
            hue   = (phase / (2 * np.pi)) % 1.0
            color = mcolors.hsv_to_rgb([hue, 0.75, 0.92])

            rect = plt.Rectangle(
                [j - size / 2, (n - 1 - i) - size / 2],
                size, size,
                color=color, zorder=3,
            )
            ax.add_patch(rect)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(BASIS_LABELS, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(BASIS_LABELS[::-1], fontsize=9)

    # Grid
    for k in np.arange(-0.5, n, 1):
        ax.axhline(k, color='#CCCCCC', lw=0.6, zorder=1)
        ax.axvline(k, color='#CCCCCC', lw=0.6, zorder=1)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_facecolor('#1A1A2E')
    fig.patch.set_facecolor('white')

    # Colourbar (phase wheel)
    sm = plt.cm.ScalarMappable(
        cmap=cm.get_cmap('hsv'),
        norm=plt.Normalize(-np.pi, np.pi),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Phase angle (rad)', fontsize=8)
    cb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Entanglement evolution
# ============================================================================

def plot_entanglement_evolution(
    t: np.ndarray,
    ent: Dict[str, np.ndarray],
    title: str = "Two-Qubit Entanglement Dynamics",
    show_purity: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot concurrence, entanglement entropy, and optionally purity vs time.

    Parameters
    ----------
    t          : (N,) time axis
    ent        : dict from track_entanglement() with keys
                 'concurrence', 'entropy_A', 'entanglement_of_formation',
                 'purity_A', 'purity_B'
    title      : figure title
    show_purity: include a purity panel
    save_path  : save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_rows = 2 + int(show_purity)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.0 * n_rows),
                              sharex=True)

    # --- Concurrence ---------------------------------------------------------
    ax = axes[0]
    ax.plot(t, ent['concurrence'], color=PALETTE['concurrence'],
            lw=2.2, label='Concurrence  C(ρ)')
    if 'entanglement_of_formation' in ent:
        ax.plot(t, ent['entanglement_of_formation'],
                '--', color=PALETTE['fidelity'], lw=1.6,
                label='Entanglement of Formation  E_F')
    ax.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel('Concurrence', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')

    # --- Entanglement entropy -----------------------------------------------
    ax = axes[1]
    ax.plot(t, ent['entropy_A'], color=PALETTE['entropy'],
            lw=2.2, label='Entanglement entropy  S(ρ_A)')
    ax.fill_between(t, ent['entropy_A'], alpha=0.12,
                    color=PALETTE['entropy'])
    ax.axhline(1.0, color=PALETTE['entropy'], lw=0.8, ls=':', alpha=0.5,
               label='Max (1 ebit)')
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel('Entropy (ebits)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')

    # --- Purity -------------------------------------------------------------
    if show_purity:
        ax = axes[2]
        ax.plot(t, ent['purity_A'], color=PALETTE['purity'],
                lw=2.0, label='Purity  Tr(ρ_A²)')
        ax.plot(t, ent['purity_B'], '--', color='#888888',
                lw=1.6, label='Purity  Tr(ρ_B²)')
        ax.axhline(0.5, ls=':', color='#333333', lw=0.8, alpha=0.5,
                   label='Max mixed (0.5)')
        ax.axhline(1.0, ls=':', color='#333333', lw=0.8, alpha=0.5,
                   label='Pure (1.0)')
        ax.set_ylim(0.3, 1.08)
        ax.set_ylabel('Purity', fontsize=10)
        ax.legend(fontsize=8, loc='lower left', framealpha=0.85)
        ax.grid(True, ls='--', alpha=0.3)
        ax.set_facecolor('#F9F9F9')

    axes[-1].set_xlabel('Time', fontsize=10)
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Side-by-side Bloch spheres for the two reduced states
# ============================================================================

def _draw_bloch(ax, r: np.ndarray, label: str, color: str = '#E63946'):
    """Draw a single Bloch sphere with one state vector."""
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='#DDDDDD', alpha=0.07,
                    linewidth=0, zorder=0)

    phi = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(phi), np.sin(phi), np.zeros(100),
            color='#AAAAAA', lw=0.6, alpha=0.5)
    ax.plot(np.zeros(100), np.cos(phi), np.sin(phi),
            color='#AAAAAA', lw=0.6, alpha=0.5)

    for (dx, dy, dz), lbl, c in [
        ((1.4, 0, 0), 'x', '#E63946'),
        ((0, 1.4, 0), 'y', '#2C7BB6'),
        ((0, 0, 1.4), 'z', '#6A994E'),
    ]:
        ax.quiver(0, 0, 0, dx, dy, dz, color=c, lw=1.2,
                  arrow_length_ratio=0.12, alpha=0.8)
        ax.text(dx*1.05, dy*1.05, dz*1.05, lbl, fontsize=9, color=c)

    rx, ry, rz = r
    rnorm = np.linalg.norm(r)

    # Draw state vector
    ax.quiver(0, 0, 0, rx, ry, rz, color=color, lw=2.5,
              arrow_length_ratio=0.15, zorder=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_title(f'{label}  (|r|={rnorm:.3f})', fontsize=10,
                 fontweight='bold', pad=8)
    ax.set_axis_off()
    ax.text(0, 0, 1.25, '|0⟩', ha='center', fontsize=8, color='#555555')
    ax.text(0, 0, -1.25, '|1⟩', ha='center', fontsize=8, color='#555555')


def plot_bloch_pair(
    rho: np.ndarray,
    title: str = "Reduced State Bloch Spheres",
    colors: Tuple[str, str] = ('#E63946', '#2C7BB6'),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Draw the reduced single-qubit states of a two-qubit density matrix.

    Traces out each qubit in turn and plots the resulting (possibly mixed)
    single-qubit state as a Bloch vector.  Mixed states appear as vectors
    with |r| < 1 (inside the sphere).

    For a maximally entangled Bell state:
        ρ_A = ρ_B = I/2  →  r = (0,0,0)  (centre of sphere)

    Parameters
    ----------
    rho    : (4,4) two-qubit density matrix
    title  : figure title
    colors : (color_A, color_B) for each Bloch vector
    save_path : save path

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .entanglement import partial_trace
    from ..density_matrix import dm_to_bloch  # reuse existing helper

    rho_A = partial_trace(rho, keep=0)
    rho_B = partial_trace(rho, keep=1)

    # Bloch vector from reduced DM
    sx2 = np.array([[0, 1], [1, 0]], dtype=complex)
    sy2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz2 = np.array([[1, 0], [0, -1]], dtype=complex)

    def _bloch(rho2):
        return np.array([
            float(np.real(np.trace(rho2 @ sx2))),
            float(np.real(np.trace(rho2 @ sy2))),
            float(np.real(np.trace(rho2 @ sz2))),
        ])

    r_A = _bloch(rho_A)
    r_B = _bloch(rho_B)

    fig = plt.figure(figsize=(11, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)

    for idx, (r, label, color) in enumerate([
        (r_A, 'Qubit A  (ρ_A = Tr_B[ρ])', colors[0]),
        (r_B, 'Qubit B  (ρ_B = Tr_A[ρ])', colors[1]),
    ]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        _draw_bloch(ax, r, label, color)
        ax.view_init(elev=20, azim=-55)

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Bell state fidelity bar chart
# ============================================================================

def plot_bell_fidelities(
    rho: np.ndarray,
    title: str = "Bell State Fidelities",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart showing fidelity of ρ with each of the four Bell states.

    F(ρ, |Φ⟩⟨Φ|) = ⟨Φ|ρ|Φ⟩  (for pure Bell state targets).
    Fidelities sum to 1 iff ρ is maximally mixed (I/4).

    Parameters
    ----------
    rho : (4,4) density matrix

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .states import bell_state

    names = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
    latex = [r'$|\Phi^+\rangle$', r'$|\Phi^-\rangle$',
             r'$|\Psi^+\rangle$', r'$|\Psi^-\rangle$']
    fids  = []
    for nm in names:
        bket = bell_state(nm)
        brho = np.outer(bket, bket.conj())
        fids.append(float(np.real(np.trace(brho @ rho))))

    colors_bar = ['#E63946', '#2C7BB6', '#6A994E', '#FF9F1C']
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(latex, fids, color=colors_bar, edgecolor='white',
                  linewidth=1.5, width=0.55, zorder=3)

    for bar, f in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width() / 2, f + 0.012,
                f'{f:.3f}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.axhline(0.25, ls='--', color='#888888', lw=1.0, alpha=0.6,
               label='Maximally mixed (0.25)')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Fidelity  F(ρ, |Bell⟩)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, axis='y', ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Entanglement sudden death figure
# ============================================================================

def plot_esd_demo(
    t: np.ndarray,
    concurrence: np.ndarray,
    purity_A: np.ndarray,
    T1_A: float,
    T1_B: float,
    title: str = "Entanglement Sudden Death",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot concurrence and single-qubit purity to show sudden death.

    The key visual: concurrence reaches zero at finite t_ESD while
    purity is still > 0.5 (qubit still has non-trivial coherence).

    Parameters
    ----------
    t           : time axis
    concurrence : concurrence trajectory C(t)
    purity_A    : single-qubit purity Tr(ρ_A²)
    T1_A, T1_B  : for annotating the T1 timescales
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)

    # Top: concurrence
    ax = axes[0]
    ax.plot(t, concurrence, color=PALETTE['concurrence'], lw=2.5,
            label='Concurrence  C(t)')

    # Find ESD time (first zero crossing)
    esd_idx = np.argmax(concurrence < 1e-4)
    if concurrence[esd_idx] < 1e-4 and esd_idx > 0:
        t_esd = t[esd_idx]
        ax.axvline(t_esd, color=PALETTE['concurrence'], ls='--', lw=1.2,
                   alpha=0.7)
        ax.annotate(
            f't_ESD ≈ {t_esd:.2f}',
            xy=(t_esd, 0.02),
            xytext=(t_esd + (t[-1]-t[0])*0.05, 0.15),
            fontsize=9, color=PALETTE['concurrence'],
            arrowprops=dict(arrowstyle='->', color=PALETTE['concurrence'],
                            lw=0.9),
        )

    ax.fill_between(t, concurrence, alpha=0.12, color=PALETTE['concurrence'])
    ax.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel('Concurrence', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')

    # Bottom: purity
    ax = axes[1]
    ax.plot(t, purity_A, color=PALETTE['purity'], lw=2.2,
            label='Purity  Tr(ρ_A²)')

    # Overlay T1 exponential decay of purity (single-qubit amplitude damping)
    T1_mean = (T1_A + T1_B) / 2.0
    purity_env = 0.5 + 0.5 * np.exp(-t / T1_mean)
    ax.plot(t, purity_env, '--', color='#888888', lw=1.4,
            label=f'½ + ½·exp(-t/T₁)  T₁={T1_mean:.1f}', alpha=0.7)

    ax.axhline(0.5, ls=':', color='#333333', lw=0.8, alpha=0.5,
               label='Fully mixed (0.5)')
    ax.axhline(1.0, ls=':', color='#333333', lw=0.8, alpha=0.5)

    ax.set_ylim(0.35, 1.08)
    ax.set_ylabel('Qubit A Purity', fontsize=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')

    fig.patch.set_facecolor('white')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Density matrix snapshot grid
# ============================================================================

def plot_dm_snapshots(
    t: np.ndarray,
    rho_t: np.ndarray,
    n_snapshots: int = 6,
    title: str = "Density Matrix Evolution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grid of |ρ| heatmaps at evenly-spaced time points.

    Parameters
    ----------
    t           : (T,) time axis
    rho_t       : (T, 4, 4) density matrix trajectory
    n_snapshots : number of panels (default 6)
    title       : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    indices = np.linspace(0, len(t) - 1, n_snapshots, dtype=int)
    fig, axes = plt.subplots(1, n_snapshots,
                              figsize=(2.8 * n_snapshots, 3.2))
    if n_snapshots == 1:
        axes = [axes]

    vmax = 1.0
    cmap = _CMAP_ABS

    for ax, idx in zip(axes, indices):
        rho  = rho_t[idx]
        data = np.abs(rho)
        im   = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax,
                         interpolation='nearest', aspect='equal')

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(BASIS_LABELS, fontsize=6, rotation=45, ha='right')
        ax.set_yticklabels(BASIS_LABELS, fontsize=6)
        ax.set_title(f't={t[idx]:.2f}', fontsize=8, fontweight='bold')
        ax.set_facecolor('#F5F5F5')

    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.02)

    # Single shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='|ρ_ij|')

    fig.patch.set_facecolor('white')
    fig.tight_layout(rect=[0, 0, 0.91, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ============================================================================
# Concurrence sweep vs circuit angle
# ============================================================================

def plot_concurrence_vs_angle(
    angles: np.ndarray,
    concurrences: np.ndarray,
    xlabel: str = "Gate angle θ (rad)",
    title: str = "Concurrence vs Gate Angle",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot concurrence as a function of a swept circuit parameter.

    Useful for showing how an XX(θ) or Ry(θ) sweep creates a sinusoidal
    pattern in entanglement.

    Parameters
    ----------
    angles      : (N,) parameter sweep
    concurrences: (N,) concurrence at each angle

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(angles, concurrences, color=PALETTE['concurrence'],
            lw=2.5, zorder=3)
    ax.fill_between(angles, concurrences, alpha=0.12,
                    color=PALETTE['concurrence'])

    # Mark maximum
    idx_max = np.argmax(concurrences)
    ax.scatter([angles[idx_max]], [concurrences[idx_max]],
               s=80, color=PALETTE['concurrence'], zorder=5,
               label=f'Max C={concurrences[idx_max]:.3f} at θ={angles[idx_max]:.3f}')

    ax.axhline(0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Concurrence  C', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
