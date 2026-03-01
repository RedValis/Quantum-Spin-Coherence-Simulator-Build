"""
examples/run_prototype5.py
==========================
Prototype 5: inhomogeneous spin ensemble — FID vs Hahn echo.

Outputs:
  prototype5_fid_sigma_sweep.png   - FID for multiple sigma values
  prototype5_fid_vs_echo.png       - FID vs echo at fixed sigma (T2* vs T2)
  prototype5_echo_sweep.png        - echo amplitude vs 2τ for two sigmas
  prototype5_bloch_sphere_fid.png  - Bloch sphere showing dephasing fan

Usage:
    python examples/run_prototype5.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ensemble import (
    simulate_ensemble_FID,
    simulate_ensemble_hahn_echo,
    sweep_ensemble_echo,
)
from src.sequences import measure_echo_amplitude

OUT    = os.path.dirname(__file__)
gamma  = 1.0
omega0 = 2 * np.pi * 0.5      # rad/µs
M0     = 1.0
T1     = 50.0
T2     = 10.0
dt     = 0.05
N      = 300
SEED   = 42

print("=== Prototype 5 - Inhomogeneous Ensemble ===\n")

# -- Plot 1: FID for several sigma values -------------------------------------
sigmas  = [0.0, 0.15, 0.30, 0.60]
colors  = ["#2C7BB6", "#6A994E", "#E63946", "#9B2226"]
t_max_fid = 3 * T2

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle(r"FID vs Inhomogeneous Broadening ($\sigma$)", fontsize=13, fontweight="bold")

ax_raw, ax_env = axes1

for sig, col in zip(sigmas, colors):
    t_f, Mx_f, My_f, _ = simulate_ensemble_FID(
        omega0=omega0, sigma=sig, N=N, T1=T1, T2=T2, M0=M0,
        t_max=t_max_fid, dt=dt, gamma=gamma, seed=SEED)
    M_perp = np.sqrt(Mx_f**2 + My_f**2)
    lbl = rf"$\sigma$={sig} rad/µs"
    ax_raw.plot(t_f, Mx_f,    color=col, lw=1.5, alpha=0.8, label=lbl)
    ax_env.plot(t_f, M_perp, color=col, lw=2.0, label=lbl)

    # overlay analytic envelope
    if sig > 0:
        env = M0 * np.exp(-t_f/T2) * np.exp(-0.5*sig**2*t_f**2)
        ax_env.plot(t_f, env, "--", color=col, lw=1.0, alpha=0.5)

# single-spin T2 envelope
t_ref = np.linspace(0, t_max_fid, 300)
ax_env.plot(t_ref, M0*np.exp(-t_ref/T2), "k--", lw=1.2, alpha=0.5,
            label=rf"$e^{{-t/T_2}}$ (single spin)")

for ax, ttl, yl in [
    (ax_raw, r"$\langle M_x(t) \rangle$",          r"$\langle M_x \rangle$"),
    (ax_env, r"Transverse magnitude $|\langle M_\perp\rangle|$", r"$|\langle M_\perp\rangle|$"),
]:
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("Time (µs)", fontsize=11)
    ax.set_ylabel(yl, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_facecolor("#F9F9F9")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)

fig1.patch.set_facecolor("white")
fig1.tight_layout()
p1 = os.path.join(OUT, "prototype5_fid_sigma_sweep.png")
fig1.savefig(p1, dpi=150, bbox_inches="tight")
print(f"  Saved → {p1}")

# -- Plot 2: FID vs Hahn echo — T2* vs T2 -------------------------------------
sigma2 = 0.40
tau2   = 6.0

t_fid2, Mx_fid2, My_fid2, _ = simulate_ensemble_FID(
    omega0=omega0, sigma=sigma2, N=N, T1=T1, T2=T2, M0=M0,
    t_max=2*tau2, dt=dt, gamma=gamma, seed=SEED)
M_perp_fid2 = np.sqrt(Mx_fid2**2 + My_fid2**2)

t_echo2, Mx_echo2, My_echo2, _ = simulate_ensemble_hahn_echo(
    omega0=omega0, sigma=sigma2, N=N, T1=T1, T2=T2, M0=M0,
    tau=tau2, dt=dt, gamma=gamma, seed=SEED)
M_perp_echo2 = np.sqrt(Mx_echo2**2 + My_echo2**2)

amp_fid2  = M_perp_fid2[np.argmin(np.abs(t_fid2 - 2*tau2))]
amp_echo2 = measure_echo_amplitude(Mx_echo2, My_echo2, t_echo2, 2*tau2)

print(f"\n  FID vs Echo  (σ={sigma2}, τ={tau2} µs, 2τ={2*tau2} µs)")
print(f"  FID  |M_perp| at 2τ   = {amp_fid2:.4f}  "
      f"(theory {M0*np.exp(-2*tau2/T2)*np.exp(-0.5*sigma2**2*(2*tau2)**2):.4f})")
print(f"  Echo amplitude at 2τ  = {amp_echo2:.4f}  "
      f"(theory {M0*np.exp(-2*tau2/T2):.4f})")
print(f"  Echo/FID ratio        = {amp_echo2/max(amp_fid2, 1e-6):.2f}x refocusing\n")

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(t_fid2, M_perp_fid2, color="#E63946", lw=2.0,
         label=rf"FID  $|\langle M_\perp\rangle|$  ($\sigma$={sigma2} rad/µs)")
ax2.plot(t_echo2, M_perp_echo2, color="#2C7BB6", lw=2.0,
         label=rf"Hahn Echo  $|\langle M_\perp\rangle|$  ($\sigma$={sigma2} rad/µs)")

t_ref2 = np.linspace(0, 2*tau2, 400)
ax2.plot(t_ref2, M0*np.exp(-t_ref2/T2), "k--", lw=1.2, alpha=0.55,
         label=rf"$e^{{-t/T_2}}$  ($T_2$={T2} µs)")
fid_env = M0 * np.exp(-t_ref2/T2) * np.exp(-0.5*sigma2**2*t_ref2**2)
ax2.plot(t_ref2, fid_env, ":", color="#E63946", lw=1.2, alpha=0.7,
         label=r"FID analytic envelope")

ax2.axvline(tau2,   color="#9B2226", lw=1.2, ls=":", alpha=0.7, label="π pulse")
ax2.axvline(2*tau2, color="#457B9D", lw=1.2, ls="--", alpha=0.9, label="echo time")
ax2.scatter([2*tau2], [amp_echo2], s=60, color="#2C7BB6", zorder=8)
ax2.scatter([2*tau2], [amp_fid2],  s=60, color="#E63946", zorder=8)

ax2.set_xlabel("Time (µs)", fontsize=12)
ax2.set_ylabel(r"$|\langle M_\perp\rangle|$", fontsize=12)
ax2.set_title(
    rf"FID vs Hahn Echo — $T_2^*$ vs $T_2$  ($\sigma$={sigma2} rad/µs, $T_2$={T2} µs)",
    fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(True, ls="--", alpha=0.35)
ax2.set_facecolor("#F9F9F9")
fig2.patch.set_facecolor("white")
fig2.tight_layout()
p2 = os.path.join(OUT, "prototype5_fid_vs_echo.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"  Saved → {p2}")

# -- Plot 3: Echo sweep — T2 extracted independent of sigma -------------------
tau_arr = np.linspace(1.0, 20.0, 14)
fig3, ax3 = plt.subplots(figsize=(8, 5))

for sig3, col3, ls3 in [(0.0, "#2C7BB6", "-"), (0.5, "#E63946", "--")]:
    two_tau3, amps3 = sweep_ensemble_echo(
        omega0=omega0, sigma=sig3, N=N, T1=T1, T2=T2, M0=M0,
        tau_values=tau_arr, dt=dt, gamma=gamma, seed=SEED)
    log_A   = np.log(np.clip(amps3, 1e-10, None))
    coeffs  = np.polyfit(two_tau3, log_A, 1)
    T2_fit  = -1.0 / coeffs[0]
    ax3.scatter(two_tau3, amps3, s=45, color=col3, zorder=5)
    ax3.plot(two_tau3, M0*np.exp(coeffs[0]*two_tau3 + coeffs[1]),
             ls3, color=col3, lw=1.8,
             label=rf"$\sigma$={sig3} rad/µs  →  $T_2^{{\rm fit}}$={T2_fit:.1f} µs")
    print(f"  sigma={sig3}: fitted T2 = {T2_fit:.2f} µs  (true={T2} µs)")

t_ref3 = np.linspace(0, 2*tau_arr[-1], 300)
ax3.plot(t_ref3, M0*np.exp(-t_ref3/T2), "k:", lw=1.5, alpha=0.5,
         label=rf"True $e^{{-2\tau/T_2}}$  ($T_2$={T2} µs)")
ax3.axhline(1/np.e, color="#888888", lw=1.0, ls="--", alpha=0.4, label="1/e")
ax3.set_xlabel(r"Echo time $2\tau$ (µs)", fontsize=12)
ax3.set_ylabel("Echo amplitude", fontsize=12)
ax3.set_title(r"Echo Amplitude vs $2\tau$ — $T_2$ independent of $\sigma$",
              fontsize=12, fontweight="bold")
ax3.legend(fontsize=9, framealpha=0.85)
ax3.grid(True, ls="--", alpha=0.35)
ax3.set_facecolor("#F9F9F9")
fig3.patch.set_facecolor("white")
fig3.tight_layout()
p3 = os.path.join(OUT, "prototype5_echo_sweep.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"\n  Saved → {p3}")
print("\nDone.")