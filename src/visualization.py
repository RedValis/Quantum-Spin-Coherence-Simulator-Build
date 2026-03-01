"""
visualization.py - Advanced visualisations for spin dynamics.
=============================================================

Prototype 2 features:
  - plot_bloch_sphere_trajectory : static 3-D Bloch sphere with spin path
  - animate_bloch_trajectory     : frame-by-frame animation (GIF / MP4)

The Bloch sphere maps the two-level quantum state to the unit sphere:
  - North pole  (+z) : |↑⟩  (spin up / ground state)
  - South pole  (-z) : |↓⟩  (spin down / excited state)
  - Equator         : superposition (transverse magnetisation)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3D
from typing import Optional


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _draw_sphere_wireframe(ax, alpha: float = 0.08, color: str = "#AAAAAA") -> None:
    """Render a translucent unit-sphere wireframe on *ax*."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, zorder=0)
    # equatorial circle
    phi = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(phi), np.sin(phi), np.zeros_like(phi),
            color="#999999", lw=0.8, alpha=0.5)


def _draw_axes(ax) -> None:
    """Draw +x, +y, +z axis arrows and labels."""
    lim = 1.35
    for (xyz, label, c) in [
        ((lim, 0, 0), "x", "#E63946"),
        ((0, lim, 0), "y", "#2C7BB6"),
        ((0, 0, lim), "z", "#6A994E"),
    ]:
        ax.quiver(0, 0, 0, *xyz, color=c, lw=1.5, arrow_length_ratio=0.12, alpha=0.9)
        ax.text(*xyz, f"  {label}", fontsize=11, color=c, fontweight="bold")

    # north / south pole labels
    ax.text(0, 0,  1.12, r"$|{\uparrow}\rangle$", ha="center", fontsize=9, color="#555555")
    ax.text(0, 0, -1.12, r"$|{\downarrow}\rangle$", ha="center", fontsize=9, color="#555555")


# ---------------------------------------------------------------------------
# Static 3-D trajectory
# ---------------------------------------------------------------------------

def plot_bloch_sphere_trajectory(
    Mx: np.ndarray,
    My: np.ndarray,
    Mz: np.ndarray,
    title: str = "Bloch Sphere Trajectory",
    color_by_time: bool = True,
    save_path: Optional[str] = None,
    elev: float = 22,
    azim: float = -55,
) -> plt.Figure:
    """Render the spin trajectory on a 3-D Bloch sphere.

    The path is colour-mapped from bright (early) to faded (late) to convey
    the passage of time.  An arrow shows the initial spin orientation.

    Parameters
    ----------
    Mx, My, Mz     : Bloch vector components from core.bloch_precession()
    title          : Figure title
    color_by_time  : Gradient colour along trajectory (True) or solid (False)
    save_path      : Save PNG to this path if given
    elev, azim     : 3-D viewing angle in degrees

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(8, 7.5))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_sphere_wireframe(ax)
    _draw_axes(ax)

    n = len(Mx)

    if color_by_time:
        # Segment the path, coloured by time fraction
        from matplotlib.collections import LineCollection
        import matplotlib.cm as cm

        cmap   = cm.get_cmap("plasma_r")
        norm   = plt.Normalize(0, n - 1)
        points = np.array([Mx, My, Mz]).T          # (N, 3)

        for i in range(n - 1):
            frac = i / (n - 1)
            col  = cmap(frac)
            ax.plot(Mx[i:i+2], My[i:i+2], Mz[i:i+2],
                    color=col, lw=1.6, alpha=max(0.35, frac))

        # colourbar (time fraction)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.05, aspect=18)
        cb.set_label("Time  (normalised)", fontsize=9)
    else:
        ax.plot(Mx, My, Mz, color="#E63946", lw=1.8, alpha=0.85)

    # initial vector arrow
    ax.quiver(0, 0, 0, Mx[0], My[0], Mz[0],
              color="#FF9F1C", lw=2.5, arrow_length_ratio=0.12,
              label=r"$\mathbf{M}(0)$", zorder=10)

    # final dot
    ax.scatter([Mx[-1]], [My[-1]], [Mz[-1]],
               s=40, color="#1A1A2E", zorder=11, label=r"$\mathbf{M}(t_{end})$")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("Mx", labelpad=6)
    ax.set_ylabel("My", labelpad=6)
    ax.set_zlabel("Mz", labelpad=6)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_facecolor("#F0F0F0")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_bloch_trajectory(
    Mx: np.ndarray,
    My: np.ndarray,
    Mz: np.ndarray,
    stride: int = 4,
    interval: int = 30,
    title: str = "Bloch Vector Animation",
    save_path: Optional[str] = None,
    writer: str = "pillow",
) -> animation.FuncAnimation:
    """Animate the spin vector sweeping across the Bloch sphere.

    A red arrow tracks the instantaneous Bloch vector, and a fading trail
    shows the recent history of the trajectory.

    Parameters
    ----------
    Mx, My, Mz : Bloch components from core.bloch_precession()
    stride     : Downsample factor — render every Nth point (default 4)
    interval   : Milliseconds between frames (default 30)
    title      : Animation title
    save_path  : If given, save as GIF (writer='pillow') or MP4 ('ffmpeg')
    writer     : 'pillow' (GIF, no extra install) or 'ffmpeg' (MP4)

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Usage
    -----
    >>> anim = animate_bloch_trajectory(Mx, My, Mz, save_path="bloch.gif")
    """
    # Downsample for smooth animation without excess frames
    Mx = Mx[::stride]
    My = My[::stride]
    Mz = Mz[::stride]
    n  = len(Mx)
    trail_len = max(1, n // 8)   # show last ~12.5% of path as trail

    fig = plt.figure(figsize=(7, 6.5))
    ax  = fig.add_subplot(111, projection="3d")

    _draw_sphere_wireframe(ax, alpha=0.07)
    _draw_axes(ax)

    # Objects we'll update each frame
    (arrow,)  = [ax.quiver(0, 0, 0, Mx[0], My[0], Mz[0],
                           color="#E63946", lw=2.8, arrow_length_ratio=0.14)]
    trail_line, = ax.plot([], [], [], color="#FFAA00", lw=1.4, alpha=0.6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("Mx"); ax.set_ylabel("My"); ax.set_zlabel("Mz")
    ax.set_title(title, fontsize=12, fontweight="bold")
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

    def _init():
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        time_text.set_text("")
        return trail_line, time_text

    def _update(frame):
        nonlocal arrow
        # Remove old arrow and redraw (quiver doesn't support set_* easily)
        arrow.remove()
        arrow = ax.quiver(0, 0, 0, Mx[frame], My[frame], Mz[frame],
                          color="#E63946", lw=2.8, arrow_length_ratio=0.14, zorder=10)

        # Trail (recent history)
        start = max(0, frame - trail_len)
        trail_line.set_data(Mx[start:frame+1], My[start:frame+1])
        trail_line.set_3d_properties(Mz[start:frame+1])

        time_text.set_text(f"frame {frame+1}/{n}")
        return trail_line, time_text

    anim = animation.FuncAnimation(
        fig, _update, frames=n,
        init_func=_init, interval=interval, blit=False,
    )

    if save_path:
        anim.save(save_path, writer=writer, fps=1000 // interval, dpi=120)

    return anim