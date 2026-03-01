"""
examples/run_prototype4.py
==========================
Demonstrates Prototype 4: pulse sequences — Hahn echo, CPMG, echo sweeps.

Generates four output files:
  1. prototype4_hahn_echo.png       - single Hahn echo waveform
  2. prototype4_echo_sweep.png      - echo amplitude vs 2τ (T2 decay curve)
  3. prototype4_cpmg.png            - CPMG multi-echo train
  4. prototype4_bloch_sphere_echo.png - Bloch sphere showing refocusing

Usage:
    python examples/run_prototype4.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sequences import (
    hahn_echo_sequence, cpmg_sequence,
    measure_echo_amplitude, sweep_echo_amplitude,
)
from src.visualization import plot_bloch_sphere_trajectory

OUT    = os.path.dirname(__file__)
gamma  = 1.0
B0     = 2 * np.pi * 0.5
B      = np.array([0., 0., B0])
M0     = 1.0
T1     = 50.0
T2     = 10.0
dt     = 0.02

print("=== Prototype 4 - Pulse Sequences (Hahn Echo + CPMG) ===\n")

# -----------------------------------------------------------------------------
# Plot 1: Single Hahn echo waveform
# -----------------------------------------------------------------------------
tau = 5.0
t_h, Mx_h, My_h, Mz_h = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau, dt=dt)

M_perp_h = np.sqrt(Mx_h**2 + My_h**2)
echo_amp  = measure_echo_amplitude(Mx_h, My_h, t_h, echo_time=2*tau)
expected  = M0 * np.exp(-2*tau/T2)

print(f"  Hahn echo:  tau={tau} µs,  2τ={2*tau} µs")
print(f"  Echo amplitude  = {echo_amp:.6f}")
print(f"  Expected M0·exp(-2τ/T2) = {expected:.6f}")
print(f"  Error           = {abs(echo_amp-expected):.2e}\n")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle(
    rf"Hahn Echo Sequence  ($\tau$={tau} µs, $T_2$={T2} µs, $T_1$={T1} µs)",
    fontsize=13, fontweight="bold")

for ax, data, color, label in zip(
    axes,
    [Mx_h, My_h, Mz_h],
    ["#E63946", "#2C7BB6", "#6A994E"],
    [r"$M_x(t)$", r"$M_y(t)$", r"$M_z(t)$"]
):
    ax.plot(t_h, data, color=color, lw=1.8, label=label)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_ylabel(label, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, ls="--", alpha=0.35)
    ax.set_facecolor("#F9F9F9")

# annotate pulse moments and echo
for ax in axes:
    ax.axvline(0,   color="#FF9F1C", lw=1.5, ls=":", alpha=0.8, label="π/2 pulse")
    ax.axvline(tau, color="#9B2226", lw=1.5, ls=":", alpha=0.8, label="π pulse")
    ax.axvline(2*tau, color="#457B9D", lw=1.5, ls="--", alpha=0.9, label="echo")

axes[0].legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("Time (µs)", fontsize=11)

# overlay |M_perp| on Mx panel
axes[0].plot(t_h, M_perp_h, color="#888888", lw=1.2, ls="--",
             alpha=0.7, label=r"$|M_\perp|$")

fig.patch.set_facecolor("white")
fig.tight_layout()
p1 = os.path.join(OUT, "prototype4_hahn_echo.png")
fig.savefig(p1, dpi=150, bbox_inches="tight")
print(f"  Saved → {p1}")

# -----------------------------------------------------------------------------
# Plot 2: Echo sweep — amplitude vs 2τ
# -----------------------------------------------------------------------------
tau_values  = np.linspace(0.5, 25.0, 30)
two_tau, amps = sweep_echo_amplitude(gamma, B, T1, T2, M0, tau_values, dt=dt)
theory_amps   = M0 * np.exp(-two_tau / T2)

# Fit T2 from the sweep data
log_A  = np.log(np.clip(amps, 1e-10, None))
coeffs = np.polyfit(two_tau, log_A, 1)
T2_fit = -1.0 / coeffs[0]
print(f"  Echo sweep: {len(tau_values)} τ values,  fitted T2 = {T2_fit:.3f} µs  (true = {T2} µs)")

fig2, ax2 = plt.subplots(figsize=(8, 4.8))
ax2.scatter(two_tau, amps, s=45, color="#E63946", zorder=5,
            label=f"Simulated echo amplitudes (N={len(tau_values)})")
ax2.plot(two_tau, theory_amps, "--", color="#2C7BB6", lw=2.0,
         label=rf"$M_0\,e^{{-2\tau/T_2}}$  ($T_2$={T2} µs)")
ax2.plot(two_tau, M0*np.exp(coeffs[0]*two_tau + coeffs[1]),
         ":", color="#6A994E", lw=1.8, alpha=0.9,
         label=rf"Fitted  $T_2^{{fit}}$ = {T2_fit:.2f} µs")
ax2.axhline(1/np.e, color="#888888", lw=1.0, ls="--", alpha=0.6,
            label=r"$1/e$")
ax2.set_xlabel(r"Echo time  $2\tau$  (µs)", fontsize=12)
ax2.set_ylabel("Echo amplitude", fontsize=12)
ax2.set_title(rf"Echo Amplitude vs $2\tau$  — Hahn Echo T₂ Measurement",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(True, ls="--", alpha=0.35)
ax2.set_facecolor("#F9F9F9")
fig2.patch.set_facecolor("white")
fig2.tight_layout()
p2 = os.path.join(OUT, "prototype4_echo_sweep.png")
fig2.savefig(p2, dpi=150, bbox_inches="tight")
print(f"  Saved → {p2}")

# -----------------------------------------------------------------------------
# Plot 3: CPMG multi-echo train
# -----------------------------------------------------------------------------
n_echoes = 8
tau_cp   = 2.0
t_cp, Mx_cp, My_cp, Mz_cp, echo_times = cpmg_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0,
    tau=tau_cp, n_echoes=n_echoes, dt=dt)

M_perp_cp = np.sqrt(Mx_cp**2 + My_cp**2)
echo_amps_cp = [measure_echo_amplitude(Mx_cp, My_cp, t_cp, et) for et in echo_times]

print(f"\n  CPMG: {n_echoes} echoes, tau={tau_cp} µs")
for i, (et, ea) in enumerate(zip(echo_times, echo_amps_cp)):
    print(f"    Echo {i+1}: t={et:.1f} µs,  amp={ea:.4f}  "
          f"(expected {M0*np.exp(-et/T2):.4f})")

fig3, (ax_wave, ax_amps) = plt.subplots(2, 1, figsize=(12, 7),
    gridspec_kw={"height_ratios": [2, 1]})
fig3.suptitle(
    rf"CPMG Echo Train  ($\tau$={tau_cp} µs, N={n_echoes}, $T_2$={T2} µs)",
    fontsize=13, fontweight="bold")

ax_wave.plot(t_cp, M_perp_cp, color="#9B2226", lw=1.6, label=r"$|M_\perp|(t)$")
ax_wave.plot(t_cp, Mz_cp,     color="#6A994E", lw=1.4, alpha=0.7,
             ls="--", label=r"$M_z(t)$")
env = M0 * np.exp(-t_cp / T2)
ax_wave.plot(t_cp, env, ":", color="#F4A261", lw=1.4, alpha=0.85,
             label=rf"$e^{{-t/T_2}}$ envelope")
for et in echo_times:
    ax_wave.axvline(et, color="#2C7BB6", lw=0.9, ls="--", alpha=0.5)
ax_wave.set_ylabel(r"Magnetisation", fontsize=11)
ax_wave.legend(fontsize=9, loc="upper right", framealpha=0.85)
ax_wave.grid(True, ls="--", alpha=0.3)
ax_wave.set_facecolor("#F9F9F9")

ax_amps.scatter(echo_times, echo_amps_cp, s=60, color="#E63946", zorder=5,
                label="Echo peaks")
ax_amps.plot(echo_times, M0*np.exp(-echo_times/T2), "--", color="#2C7BB6",
             lw=1.8, label=rf"$e^{{-t/T_2}}$  ($T_2$={T2} µs)")
ax_amps.set_xlabel("Time (µs)", fontsize=11)
ax_amps.set_ylabel("Echo amplitude", fontsize=11)
ax_amps.legend(fontsize=9, framealpha=0.85)
ax_amps.grid(True, ls="--", alpha=0.3)
ax_amps.set_facecolor("#F9F9F9")

fig3.patch.set_facecolor("white")
fig3.tight_layout()
p3 = os.path.join(OUT, "prototype4_cpmg.png")
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"\n  Saved → {p3}")

# -----------------------------------------------------------------------------
# Plot 4: Bloch sphere — Hahn echo refocusing trajectory
# -----------------------------------------------------------------------------
tau_s = 3.0
t_s, Mx_s, My_s, Mz_s = hahn_echo_sequence(
    gamma=gamma, B=B, T1=T1, T2=T2, M0=M0, tau=tau_s, dt=0.04)

p4 = os.path.join(OUT, "prototype4_bloch_sphere_echo.png")
plot_bloch_sphere_trajectory(
    Mx_s, My_s, Mz_s,
    title=rf"Bloch Sphere — Hahn Echo ($\tau$={tau_s} µs)",
    save_path=p4)
print(f"  Saved → {p4}")

print("\nDone.")