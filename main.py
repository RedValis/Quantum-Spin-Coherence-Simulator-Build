"""
app.py – Spin Coherence Simulator
Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.core import simulate_bloch, plot_bloch_relaxation
from src.sequences import (
    hahn_echo_sequence, cpmg_sequence,
    measure_echo_amplitude, sweep_echo_amplitude,
)
from src.ensemble import (
    simulate_ensemble_FID,
    simulate_ensemble_hahn_echo,
    sweep_ensemble_echo,
)
from src.fitting import (
    fit_T2_to_data, fit_multi_to_data,
    generate_synthetic_data, MODEL_REGISTRY,
)
from src.visualization import plot_bloch_sphere_trajectory


# ===========================================================================
# Page config
# ===========================================================================

st.set_page_config(
    page_title="Spin Coherence Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme detection
try:
    _theme_base = st.get_option("theme.base") or "light"
except Exception:
    _theme_base = "light"

IS_DARK = (_theme_base == "dark")

if IS_DARK:
    plt.style.use("dark_background")
    _fig_bg = "#0e1117"
    _ax_bg  = "#1a1d24"
    _grid_c = "#2d3139"
    _txt_c  = "#fafafa"
else:
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    matplotlib.use("Agg")
    _fig_bg = "white"
    _ax_bg  = "#f8f9fb"
    _grid_c = "#e0e0e0"
    _txt_c  = "#111111"

st.markdown("""
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 1rem; }
    h1 { font-size: 1.55rem; font-weight: 700; }
    .section-divider { border-top: 1px solid rgba(128,128,128,0.25);
                       margin: 0.6rem 0 0.8rem 0; }
    .result-box { border-left: 3px solid #2C7BB6; border-radius: 4px;
                  padding: 0.55rem 0.85rem; margin: 0.35rem 0;
                  font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Theme helpers
# ===========================================================================

def _style_fig(fig, *axes):
    fig.patch.set_facecolor(_fig_bg)
    for ax in axes:
        ax.set_facecolor(_ax_bg)
        ax.tick_params(colors=_txt_c, which="both")
        ax.xaxis.label.set_color(_txt_c)
        ax.yaxis.label.set_color(_txt_c)
        ax.title.set_color(_txt_c)
        ax.grid(True, ls="--", alpha=0.35, color=_grid_c)
        for spine in ax.spines.values():
            spine.set_edgecolor(_grid_c)
    fig.tight_layout()

def _close(fig):
    plt.close(fig)
    return fig

C_BLUE   = "#2C7BB6"
C_RED    = "#E63946"
C_GREEN  = "#6A994E"
C_ORANGE = "#F4A261"
C_GREY   = "#888888"

MODEL_LABELS = {
    "simple_T2":    "L(t) = exp(-t/T2)",
    "gaussian_fid": "L(t) = exp(-t/T2) * exp(-sigma^2 t^2 / 2)",
    "hahn_echo":    "A(tau) = exp(-2*tau/T2)",
    "T1_recovery":  "Mz(t) = M0 * (1 - exp(-t/T1))",
    "stretched_T2": "L(t) = exp(-(t/T2)^beta)",
}


# ===========================================================================
# Cached simulation wrappers
# ===========================================================================

@st.cache_data(show_spinner=False)
def _run_single_spin(gamma, omega0, T1, T2, t_max, dt):
    B = np.array([0.0, 0.0, omega0 / gamma])
    return simulate_bloch(M_init=[1.0, 0.0, 0.0], gamma=gamma, B=B,
                          T1=T1, T2=T2, M0=1.0, t_max=t_max, dt=dt)

@st.cache_data(show_spinner=False)
def _run_ensemble_fid(gamma, omega0, sigma, N, T1, T2, t_max, dt, seed):
    return simulate_ensemble_FID(omega0=omega0, sigma=sigma, N=N,
                                 T1=T1, T2=T2, M0=1.0, t_max=t_max,
                                 dt=dt, gamma=gamma, seed=seed)

@st.cache_data(show_spinner=False)
def _run_hahn_echo(gamma, omega0, T1, T2, tau, dt):
    B = np.array([0.0, 0.0, omega0 / gamma])
    return hahn_echo_sequence(gamma=gamma, B=B, T1=T1, T2=T2,
                              M0=1.0, tau=tau, dt=dt)

@st.cache_data(show_spinner=False)
def _run_ensemble_echo(gamma, omega0, sigma, N, T1, T2, tau, dt, seed):
    return simulate_ensemble_hahn_echo(omega0=omega0, sigma=sigma, N=N,
                                       T1=T1, T2=T2, M0=1.0, tau=tau,
                                       dt=dt, gamma=gamma, seed=seed)

@st.cache_data(show_spinner=False)
def _run_cpmg(gamma, omega0, T1, T2, tau, n_echoes, dt):
    B = np.array([0.0, 0.0, omega0 / gamma])
    return cpmg_sequence(gamma=gamma, B=B, T1=T1, T2=T2, M0=1.0,
                         tau=tau, n_echoes=n_echoes, dt=dt)

@st.cache_data(show_spinner=False)
def _run_echo_sweep(gamma, omega0, sigma, N, T1, T2,
                    tau_min, tau_max, n_tau, dt, seed, use_ensemble):
    tau_arr = np.linspace(tau_min, tau_max, n_tau)
    B = np.array([0.0, 0.0, omega0 / gamma])
    if use_ensemble and sigma > 0:
        return sweep_ensemble_echo(omega0=omega0, sigma=sigma, N=N,
                                   T1=T1, T2=T2, M0=1.0, tau_values=tau_arr,
                                   dt=dt, gamma=gamma, seed=seed)
    return sweep_echo_amplitude(gamma=gamma, B=B, T1=T1, T2=T2, M0=1.0,
                                tau_values=tau_arr, dt=dt)


# ===========================================================================
# Plot builders
# ===========================================================================

def _fig_relaxation(t, Mx, My, Mz, T1, T2, title):
    fig = plot_bloch_relaxation(t, Mx, My, Mz, T1=T1, T2=T2,
                                title=title, time_unit="us")
    _style_fig(fig, *fig.axes)
    return _close(fig)

def _fig_echo_detail(t, Mx, My, tau):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, np.sqrt(Mx**2 + My**2), color=C_BLUE, lw=2, label="|M_perp|(t)")
    ax.axvline(tau,   color=C_RED,    lw=1.2, ls=":", label="pi pulse")
    ax.axvline(2*tau, color=C_ORANGE, lw=1.4, ls="--", label="echo")
    ax.set_xlabel("Time (us)"); ax.set_ylabel("|M_perp|")
    ax.legend(fontsize=9)
    _style_fig(fig, ax)
    return _close(fig)

def _fig_echo_sweep(two_tau, amps, T2):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(two_tau, amps, s=50, color=C_BLUE, zorder=5,
               label="Simulated echo amplitudes")
    ax.plot(two_tau, np.exp(-two_tau / T2), "--", color=C_RED, lw=1.8,
            label=f"exp(-2t/T2)  T2={T2} us")
    log_A  = np.log(np.clip(amps, 1e-10, None))
    coeffs = np.polyfit(two_tau, log_A, 1)
    T2_fit = -1.0 / coeffs[0]
    ax.plot(two_tau, np.exp(coeffs[0]*two_tau + coeffs[1]),
            ":", color=C_GREEN, lw=1.6, label=f"Fitted T2 = {T2_fit:.2f} us")
    ax.axhline(1/np.e, color=C_GREY, lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Echo time 2t (us)"); ax.set_ylabel("Echo amplitude")
    ax.set_title("Echo Amplitude vs 2t", fontweight="bold")
    ax.legend(fontsize=9)
    _style_fig(fig, ax)
    return _close(fig), T2_fit

def _fig_fid(t_fid, Mx_f, My_f, sigma, T2,
             t_echo=None, Mx_e=None, My_e=None, tau=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    M_perp_f = np.sqrt(Mx_f**2 + My_f**2)
    axes[0].plot(t_fid, Mx_f, color=C_BLUE, lw=1.6, label="<Mx>")
    axes[0].plot(t_fid, My_f, color=C_RED,  lw=1.6, label="<My>", alpha=0.8)
    axes[0].axhline(0, color=_txt_c, lw=0.5, alpha=0.3)
    axes[0].set_xlabel("Time (us)"); axes[0].set_title("Transverse components")
    axes[0].legend(fontsize=9)
    axes[1].plot(t_fid, M_perp_f, color=C_BLUE, lw=2,
                 label=f"FID |<M_perp>|  sigma={sigma:.2f}")
    axes[1].plot(t_fid, np.exp(-t_fid/T2)*np.exp(-0.5*sigma**2*t_fid**2),
                 "--", color=C_BLUE, lw=1.2, alpha=0.5, label="FID analytic")
    axes[1].plot(t_fid, np.exp(-t_fid/T2), color=C_GREY, lw=1,
                 ls="--", alpha=0.6, label=f"exp(-t/T2)")
    if t_echo is not None:
        axes[1].plot(t_echo, np.sqrt(Mx_e**2 + My_e**2), color=C_RED, lw=2,
                     label=f"Echo  tau={tau} us")
        axes[1].axvline(tau,   color=C_RED,    lw=1,   ls=":", alpha=0.6)
        axes[1].axvline(2*tau, color=C_ORANGE, lw=1.2, ls="--", alpha=0.8)
    axes[1].set_xlabel("Time (us)"); axes[1].set_title("T2* vs T2")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Ensemble FID  sigma={sigma:.2f} rad/us  T2={T2} us",
                 fontsize=11, fontweight="bold", color=_txt_c)
    _style_fig(fig, *axes)
    return _close(fig)

def _fig_cpmg(t, Mx, My, Mz, echo_times, tau, T2, n_echoes):
    fig, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(12, 7),
                                      gridspec_kw={"height_ratios": [2, 1]})
    M_perp = np.sqrt(Mx**2 + My**2)
    ax_w.plot(t, M_perp, color=C_BLUE,  lw=1.6, label="|M_perp|(t)")
    ax_w.plot(t, Mz,     color=C_GREEN, lw=1.2, ls="--", alpha=0.7, label="Mz(t)")
    ax_w.plot(t, np.exp(-t/T2), ":", color=C_RED, lw=1.4, alpha=0.8,
              label=f"exp(-t/T2)  T2={T2} us")
    for et in echo_times:
        ax_w.axvline(et, color=C_GREY, lw=0.7, ls="--", alpha=0.4)
    ax_w.set_ylabel("Magnetisation"); ax_w.legend(fontsize=8)
    ax_w.set_title(f"CPMG  tau={tau} us  N={n_echoes}  T2={T2} us",
                   fontweight="bold")
    echo_amps = [measure_echo_amplitude(Mx, My, t, et) for et in echo_times]
    ax_a.scatter(echo_times, echo_amps, s=55, color=C_BLUE, zorder=5,
                 label="Echo peaks")
    ax_a.plot(echo_times, np.exp(-np.array(echo_times)/T2),
              "--", color=C_RED, lw=1.8, label=f"exp(-t/T2)")
    ax_a.set_xlabel("Time (us)"); ax_a.set_ylabel("Echo amplitude")
    ax_a.legend(fontsize=8)
    _style_fig(fig, ax_w, ax_a)
    return _close(fig)

def _fig_fit(t_data, L_data, L_fit, params, errors, model_label):
    fig, (ax_m, ax_r) = plt.subplots(2, 1, figsize=(8, 5.5),
                                      gridspec_kw={"height_ratios": [3, 1]})
    ax_m.scatter(t_data, L_data, s=30, color=C_BLUE, alpha=0.75,
                 zorder=5, label="Data")
    ax_m.plot(t_data, L_fit, color=C_RED, lw=2.2, label="Best fit")
    param_str = "    ".join(
        f"{k} = {v:.4f} +/- {errors.get(k, 0):.4f}"
        for k, v in params.items())
    ax_m.set_title(f"Model: {model_label}\n{param_str}", fontsize=9)
    ax_m.set_ylabel("Coherence"); ax_m.legend(fontsize=9)
    resid = L_data - L_fit
    w = (t_data[1] - t_data[0]) * 0.8 if len(t_data) > 1 else 0.1
    ax_r.bar(t_data, resid, width=w, color=C_GREEN, alpha=0.75)
    ax_r.axhline(0, color=_txt_c, lw=0.8)
    ax_r.set_xlabel("Time (us)"); ax_r.set_ylabel("Residual")
    _style_fig(fig, ax_m, ax_r)
    return _close(fig)


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("## Spin Coherence Simulator")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    experiment = st.selectbox(
        "Experiment type",
        ["Single Spin (Bloch)", "Ensemble FID", "Hahn Echo",
         "CPMG Train", "Echo Sweep", "FID vs Echo (T2* comparison)"],
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Physical parameters**")

    gamma  = 1.0
    f0     = st.slider("B0 frequency f0 (MHz)", 0.1, 2.0, 0.5, 0.05)
    omega0 = 2 * np.pi * f0
    T2     = st.slider("T2 (us)", 0.5, 50.0, 10.0, 0.5)
    T1     = st.slider("T1 (us)", T2, 200.0, float(max(T2*3, T2+1.0)), 1.0)

    needs_ensemble = experiment in [
        "Ensemble FID", "FID vs Echo (T2* comparison)", "Echo Sweep"]
    if needs_ensemble:
        sigma = st.slider("Frequency spread sigma (rad/us)", 0.0, 2.0, 0.3, 0.05)
        N     = st.slider("Number of spins N", 20, 300, 100, 10)
        seed  = int(st.number_input("RNG seed", 0, 9999, 42, 1))
    else:
        sigma, N, seed = 0.0, 1, 42

    needs_tau = experiment in [
        "Hahn Echo", "CPMG Train", "Echo Sweep",
        "FID vs Echo (T2* comparison)"]
    tau = (st.slider("Half-echo time tau (us)", 0.1, 30.0, 3.0, 0.1)
           if needs_tau else 3.0)
    n_echoes = (st.slider("Number of echoes", 1, 20, 8, 1)
                if experiment == "CPMG Train" else 8)

    if experiment == "Echo Sweep":
        tau_min = st.slider("tau min (us)", 0.1, 5.0,  0.5,  0.1)
        tau_max = st.slider("tau max (us)", 1.0, 50.0, 20.0, 1.0)
        n_tau   = st.slider("Number of tau points", 5, 30, 15, 1)
    else:
        tau_min, tau_max, n_tau = 0.5, 20.0, 15

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Simulation settings**")

    t_max_def = float(max(4*T2, 4*tau if needs_tau else 4*T2))
    t_max = st.slider("Duration t_max (us)", 5.0, 200.0, t_max_def, 5.0)
    dt    = st.select_slider("Time step dt (us)",
                             [0.005, 0.01, 0.02, 0.05, 0.1], value=0.05)
    show_sphere = st.checkbox("Show Bloch sphere", value=False)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    run = st.button("Simulate", type="primary", use_container_width=True)


# ===========================================================================
# Main
# ===========================================================================

st.markdown("# Spin Coherence Simulator")
st.caption("Bloch equations  |  T1/T2 relaxation  |  "
           "Pulse sequences  |  Ensemble inhomogeneity  |  Parameter fitting")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab_sim, tab_fit = st.tabs(["Simulation", "Fitting"])


# ── Simulation tab ────────────────────────────────────────────────────────────
with tab_sim:

    # KEY FIX: never call st.stop() in a tabbed app.
    # Use if/else throughout so the fitting tab always renders.
    if not run and "sim_result" not in st.session_state:
        st.info("Configure parameters in the sidebar, then press Simulate.")

    else:
        # Run simulation only when button pressed; otherwise use cached result
        if run:
            with st.spinner("Running simulation..."):
                result = {}

                if experiment == "Single Spin (Bloch)":
                    t, Mx, My, Mz = _run_single_spin(
                        gamma, omega0, T1, T2, t_max, dt)
                    result = dict(t=t, Mx=Mx, My=My, Mz=Mz,
                                  mode="single", T1=T1, T2=T2,
                                  experiment=experiment)

                elif experiment == "Ensemble FID":
                    t, Mx, My, Mz = _run_ensemble_fid(
                        gamma, omega0, sigma, N, T1, T2, t_max, dt, seed)
                    result = dict(t=t, Mx=Mx, My=My, Mz=Mz,
                                  mode="fid", T1=T1, T2=T2,
                                  sigma=sigma, experiment=experiment)

                elif experiment == "Hahn Echo":
                    t, Mx, My, Mz = _run_hahn_echo(
                        gamma, omega0, T1, T2, tau, dt)
                    result = dict(t=t, Mx=Mx, My=My, Mz=Mz,
                                  mode="echo", T1=T1, T2=T2,
                                  tau=tau, experiment=experiment)

                elif experiment == "CPMG Train":
                    t, Mx, My, Mz, echo_times = _run_cpmg(
                        gamma, omega0, T1, T2, tau, n_echoes, dt)
                    result = dict(t=t, Mx=Mx, My=My, Mz=Mz,
                                  mode="cpmg", T1=T1, T2=T2, tau=tau,
                                  n_echoes=n_echoes, echo_times=echo_times,
                                  experiment=experiment)

                elif experiment == "Echo Sweep":
                    two_tau, amps = _run_echo_sweep(
                        gamma, omega0, sigma, N, T1, T2,
                        tau_min, tau_max, n_tau, dt, seed,
                        use_ensemble=(sigma > 0))
                    result = dict(mode="sweep", T2=T2,
                                  two_tau=two_tau, amps=amps,
                                  experiment=experiment)

                else:  # FID vs Echo
                    t_fid, Mx_f, My_f, _ = _run_ensemble_fid(
                        gamma, omega0, sigma, N, T1, T2, 2*tau, dt, seed)
                    t_echo, Mx_e, My_e, _ = _run_ensemble_echo(
                        gamma, omega0, sigma, N, T1, T2, tau, dt, seed)
                    result = dict(mode="comparison", T2=T2, sigma=sigma,
                                  tau=tau, t_fid=t_fid, Mx_f=Mx_f,
                                  My_f=My_f, t_echo=t_echo, Mx_e=Mx_e,
                                  My_e=My_e, experiment=experiment)

                # Persist the entire result so tab switching doesn't lose it
                st.session_state["sim_result"] = result

        # Always render from session state
        r    = st.session_state["sim_result"]
        mode = r["mode"]
        T2_r = r["T2"]

        # Metrics
        if mode not in ("sweep", "comparison"):
            t  = r["t"]; Mx = r["Mx"]; My = r["My"]; Mz = r["Mz"]
            Mp = np.sqrt(Mx**2 + My**2)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T2 (us)", f"{T2_r:.1f}")
            c2.metric("T1 (us)", f"{r['T1']:.1f}")
            idx = np.argmin(np.abs(t - T2_r))
            c3.metric("|M_perp| at T2", f"{Mp[idx]:.4f}",
                      delta=f"expected {1/np.e:.4f}")
            if mode == "echo":
                amp = measure_echo_amplitude(Mx, My, t, 2*r["tau"])
                c4.metric("Echo amp at 2t", f"{amp:.4f}",
                          delta=f"expected {np.exp(-2*r['tau']/T2_r):.4f}")
            elif r.get("sigma", 0) > 0:
                c4.metric("T2* estimate (us)",
                          f"{1.0/(1.0/T2_r + r['sigma']):.2f}")
            else:
                c4.metric("f0 (MHz)", f"{f0:.2f}")

        # Plots
        if mode == "sweep":
            fig_sw, T2_fit_sw = _fig_echo_sweep(r["two_tau"], r["amps"], T2_r)
            st.pyplot(fig_sw)
            st.markdown(
                f'<div class="result-box">Fitted T2 = <strong>'
                f'{T2_fit_sw:.2f} us</strong>&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'Input T2 = {T2_r} us</div>',
                unsafe_allow_html=True)

        elif mode == "comparison":
            st.pyplot(_fig_fid(r["t_fid"], r["Mx_f"], r["My_f"],
                               r["sigma"], T2_r,
                               r["t_echo"], r["Mx_e"], r["My_e"], r["tau"]))
            amp_fid  = np.sqrt(r["Mx_f"]**2 + r["My_f"]**2)[
                np.argmin(np.abs(r["t_fid"] - 2*r["tau"]))]
            amp_echo = measure_echo_amplitude(
                r["Mx_e"], r["My_e"], r["t_echo"], 2*r["tau"])
            c1, c2, c3 = st.columns(3)
            c1.metric("FID |M_perp| at 2t",  f"{amp_fid:.4f}")
            c2.metric("Echo amplitude at 2t", f"{amp_echo:.4f}")
            c3.metric("Refocusing factor",
                      f"{amp_echo/max(amp_fid, 1e-6):.1f}x")

        elif mode == "cpmg":
            st.pyplot(_fig_cpmg(r["t"], r["Mx"], r["My"], r["Mz"],
                                r["echo_times"], r["tau"], T2_r, r["n_echoes"]))

        elif mode == "fid":
            st.pyplot(_fig_fid(r["t"], r["Mx"], r["My"],
                               r["sigma"], T2_r))

        else:
            st.pyplot(_fig_relaxation(r["t"], r["Mx"], r["My"], r["Mz"],
                                      r["T1"], T2_r, r["experiment"]))
            if mode == "echo":
                st.pyplot(_fig_echo_detail(r["t"], r["Mx"], r["My"], r["tau"]))

        if show_sphere and mode not in ("sweep", "comparison", "cpmg"):
            st.markdown("---")
            st.markdown("**Bloch sphere trajectory**")
            with st.spinner("Rendering..."):
                fig_sp = plot_bloch_sphere_trajectory(
                    r["Mx"], r["My"], r["Mz"], title=r["experiment"])
                _style_fig(fig_sp, *fig_sp.axes)
                st.pyplot(_close(fig_sp))

        if mode not in ("sweep", "comparison"):
            st.markdown("---")
            df_exp = pd.DataFrame({
                "t_us":   r["t"], "Mx": r["Mx"], "My": r["My"], "Mz": r["Mz"],
                "M_perp": np.sqrt(r["Mx"]**2 + r["My"]**2),
            })
            st.download_button(
                "Download simulation as CSV",
                data=df_exp.to_csv(index=False).encode(),
                file_name="spin_coherence_simulation.csv",
                mime="text/csv")


# ── Fitting tab ───────────────────────────────────────────────────────────────
with tab_fit:
    st.markdown("## Parameter Fitting")
    st.caption("Fit T2, T1, sigma, or beta to experimental or synthetic data.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_ctrl, col_data = st.columns([1, 2])

    with col_ctrl:
        data_source = st.radio(
            "Data source",
            ["Synthetic (demo)", "Upload CSV"])

        fit_model = st.selectbox(
            "Fit model",
            list(MODEL_LABELS.keys()),
            format_func=lambda k: MODEL_LABELS[k])

        st.markdown("**Synthetic data parameters**")
        true_T2_fit = st.slider("True T2 (us)", 0.5, 30.0, 8.0, 0.5)
        noise_fit   = st.slider("Noise level",  0.0,  0.10, 0.03, 0.005)
        n_pts_fit   = st.slider("Data points",  10,   100,  40,   5)
        t_fit_range = st.slider("Time range (us)", 1.0, 100.0,
                                float(min(4*true_T2_fit, 60.0)), 1.0)

        extra_params = {}
        if fit_model == "gaussian_fid":
            extra_params["sigma"] = st.slider(
                "True sigma (synthetic)", 0.05, 1.0, 0.25, 0.05)
        elif fit_model == "T1_recovery":
            extra_params["T1"] = st.slider(
                "True T1 (synthetic)", 1.0, 100.0, 25.0, 1.0)

        st.markdown('<div class="section-divider"></div>',
                    unsafe_allow_html=True)
        run_fit = st.button("Run Fit", type="primary",
                            use_container_width=True)

    with col_data:
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Two-column CSV: time, coherence", type=["csv", "txt"])
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded, header=None)
                    t_up  = df_up.iloc[:, 0].values.astype(float)
                    L_up  = df_up.iloc[:, 1].values.astype(float)
                    # Store with distinct key — never overwritten by anything else
                    st.session_state["uploaded_t"] = t_up
                    st.session_state["uploaded_L"] = L_up
                    st.success(f"Loaded {len(t_up)} data points.")
                    fig_up, ax_up = plt.subplots(figsize=(6, 3))
                    ax_up.scatter(t_up, L_up, s=20, color=C_BLUE, alpha=0.7)
                    ax_up.set_xlabel("Time"); ax_up.set_ylabel("Coherence")
                    ax_up.set_title("Uploaded data preview")
                    _style_fig(fig_up, ax_up)
                    st.pyplot(_close(fig_up))
                except Exception as e:
                    st.error(f"Could not parse file: {e}")
            elif "uploaded_t" in st.session_state:
                st.info(f"Using previously uploaded data "
                        f"({len(st.session_state['uploaded_t'])} points).")
            else:
                st.info("Upload a two-column CSV: time, coherence.")

        else:
            # Preview only — never touches session state
            t_prev = np.linspace(0, t_fit_range, n_pts_fit)
            rng    = np.random.default_rng(42)
            L_prev = (np.exp(-t_prev / true_T2_fit)
                      + rng.normal(0, noise_fit, len(t_prev)))
            fig_syn, ax_syn = plt.subplots(figsize=(6, 3))
            ax_syn.scatter(t_prev, L_prev, s=22, color=C_BLUE,
                           alpha=0.8, label="Preview (noisy)")
            ax_syn.plot(t_prev, np.exp(-t_prev/true_T2_fit), "--",
                        color=C_RED, lw=1.4, alpha=0.7,
                        label=f"True  T2={true_T2_fit} us")
            ax_syn.set_xlabel("Time (us)"); ax_syn.set_ylabel("Coherence")
            ax_syn.set_title("Synthetic data preview")
            ax_syn.legend(fontsize=9)
            _style_fig(fig_syn, ax_syn)
            st.pyplot(_close(fig_syn))

    # Run fit — generate data + fit as one atomic action, store snapshot
    if run_fit:
        with st.spinner("Fitting..."):
            try:
                if data_source == "Upload CSV":
                    if "uploaded_t" not in st.session_state:
                        st.error("No uploaded data. Please upload a CSV first.")
                    else:
                        t_fd = st.session_state["uploaded_t"]
                        L_fd = st.session_state["uploaded_L"]
                        do_fit = True
                else:
                    t_fd, L_fd = generate_synthetic_data(
                        t_max=t_fit_range, n_points=n_pts_fit,
                        T2=true_T2_fit, noise_level=noise_fit,
                        model=fit_model, seed=42, **extra_params)
                    do_fit = True

                if do_fit:
                    _, param_names, _ = MODEL_REGISTRY[fit_model]
                    if fit_model in ("simple_T2", "hahn_echo"):
                        T2_r, T2_e, L_fitted = fit_T2_to_data(
                            t_fd, L_fd,
                            initial_guess=float(true_T2_fit),
                            model=fit_model)
                        params_r = {param_names[0]: T2_r}
                        errors_r = {param_names[0]: T2_e}
                    else:
                        params_r, errors_r, L_fitted = fit_multi_to_data(
                            t_fd, L_fd, model=fit_model)

                    st.session_state["fit_result"] = {
                        "t":      t_fd,
                        "L_data": L_fd,
                        "L_fit":  L_fitted,
                        "params": params_r,
                        "errors": errors_r,
                        "model":  fit_model,
                    }

            except Exception as e:
                st.error(f"Fit failed: {e}")
                st.session_state.pop("fit_result", None)

    # Render from session state — completely decoupled from button state
    if "fit_result" in st.session_state:
        fr = st.session_state["fit_result"]
        st.markdown('<div class="section-divider"></div>',
                    unsafe_allow_html=True)
        st.markdown("**Fit result**")

        st.pyplot(_fig_fit(fr["t"], fr["L_data"], fr["L_fit"],
                           fr["params"], fr["errors"],
                           MODEL_LABELS[fr["model"]]))

        res_cols = st.columns(len(fr["params"]))
        for col, (pname, pval) in zip(res_cols, fr["params"].items()):
            col.metric(pname, f"{pval:.4f}",
                       delta=f"+/- {fr['errors'].get(pname, 0):.4f}")

        resid = fr["L_data"] - fr["L_fit"]
        st.caption(
            f"RMSE = {np.sqrt(np.mean(resid**2)):.5f}   |   "
            f"chi2 = {np.sum(resid**2):.5f}   |   "
            f"N = {len(fr['t'])} points   |   "
            f"Model: {MODEL_LABELS[fr['model']]}")

        st.download_button(
            "Download fit results as CSV",
            data=pd.DataFrame({
                "t": fr["t"], "data": fr["L_data"],
                "fit": fr["L_fit"], "residual": resid,
            }).to_csv(index=False).encode(),
            file_name="fit_results.csv",
            mime="text/csv")


st.markdown("---")
st.caption("Spin Coherence Simulator  |  "
           "Bloch equations  T1/T2  Pulse sequences  Ensemble  Fitting")