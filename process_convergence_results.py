# ITNG YMYFA 80asra YA
# IMZZ
# process_convergence_results.py

"""
Process RIS-aided cell-free convergence runs and generate IEEE-friendly figures/tables.

What this script does:
1. Reads the six paper runs from the paper_data folder beside this script.
2. Assigns each run its paper experiment tag.
3. Generates IEEE-friendly convergence plots for:
   - association game optimizer as relative EE improvement (%)
   - power allocation optimizer as normalized Dinkelbach objective alpha^(n)/alpha^(final)
4. Saves each figure as both PDF and PNG.
5. Writes CSV tables and a human-readable text report.

Expected run tags:
    final_tauK_random_S200
    final_tauKhalf_random_S200
    sen_tauK_strongest_S100
    sen_tauK_full_S100
    sen_tauKhalf_strongest_S100
    sen_tauKhalf_full_S100

Run:
    python process_convergence_results.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter


# --------------------------------------------------------
# User-facing constants
# --------------------------------------------------------

KNOWN_TAGS = [
    "final_tauK_random_S200",
    "final_tauKhalf_random_S200",
    "sen_tauK_strongest_S100",
    "sen_tauK_full_S100",
    "sen_tauKhalf_strongest_S100",
    "sen_tauKhalf_full_S100",
]

# The archive folders are named after the original paper runs. Keep these
# identifiers explicit so an unrelated experiment cannot be selected by mistake.
PAPER_RUN_FOLDERS = {
    "final_tauK_random_S200": "convergence_657555_AO_final_tauK_random_S200",
    "final_tauKhalf_random_S200": "convergence_657758_AO_final_tauKhalf_random_S200",
    "sen_tauK_strongest_S100": "convergence_657960_AO_sen_tauK_strongest_S100",
    "sen_tauK_full_S100": "convergence_658160_AO_sen_tauK_full_S100",
    "sen_tauKhalf_strongest_S100": "convergence_658360_AO_sen_tauKhalf_strongest_S100",
    "sen_tauKhalf_full_S100": "convergence_658562_AO_sen_tauKhalf_full_S100",
}

PROJECT_ROOT = Path(__file__).resolve().parent
PAPER_DATA_ROOT = PROJECT_ROOT / "paper_data"

SCENARIOS = ["no_ris", "single_ris"]
SCENARIO_LABELS = {
    "no_ris": "No RIS",
    "single_ris": "Single RIS",
}

PILOT_LABELS = {
    "tauK": r"$\tau_p=K$",
    "tauKhalf": r"$\tau_p=K/2$",
}

INIT_LABELS = {
    "random": "random",
    "strongest": "strongest singleton",
    "singleton": "strongest singleton",
    "full": "full candidate",
    "fullcandidate": "full candidate",
}

# Main random runs used for convergence figures.
MAIN_RANDOM_TAG_TAUK = "final_tauK_random_S200"
MAIN_RANDOM_TAG_TAUKHALF = "final_tauKhalf_random_S200"

# IEEE single-column figure sizes in inches.
# Side-by-side is tight in one IEEE column, so the font sizes below are intentionally compact.
FIGSIZE_SIDE_BY_SIDE = (3.5, 1.75)
FIGSIZE_TOP_BOTTOM = (3.5, 3.25)
FIGSIZE_STANDALONE = (3.5, 2.05)

DPI = 600
OUTPUT_ROOT = PROJECT_ROOT / "figures"
DEFAULT_OUTPUT_TAG = "paper_reproduction"

# Plot controls.
# Use padded means for the main paper figures. Available-only means can be biased at late
# iterations because only the slowest/non-converged setups remain in the average.
USE_PADDED_CURVES_FOR_PAPER = True

# Put the initial point at iteration 1 rather than 0.
# This keeps the x-axis labels as 1, 2, 3, ... as requested.
X_START_AT_ONE = True

# Compact IEEE-style fonts.
# These are intentionally smaller than the previous version because the figures
# will be placed as two standalone subfigures inside one IEEE figure environment.
BASE_FONT_SIZE = 4.7
AXIS_LABEL_SIZE = 4.9
TICK_LABEL_SIZE = 4.2
LEGEND_FONT_SIZE = 3.8
PANEL_LABEL_SIZE = 4.7
LINE_WIDTH = 0.95
MARKER_SIZE = 1.9

# Association-figure axis formatting.  The same range and ticks are used for
# both tau_p=K and tau_p=K/2 so the stacked subfigures are visually comparable.
ASSOCIATION_YLIM = (0.0, 17.5)
ASSOCIATION_YTICKS = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5]


# --------------------------------------------------------
# Matplotlib style
# --------------------------------------------------------

def configure_matplotlib() -> None:
    """Configure Matplotlib for IEEE/EDAS-friendly PDF output."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "lines.linewidth": LINE_WIDTH,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "mathtext.fontset": "stix",
        "font.family": "serif",
    })


# --------------------------------------------------------
# Data classes
# --------------------------------------------------------

@dataclass
class RunData:
    tag: str
    path: Path
    data: Dict[str, np.ndarray]
    pilot: str
    initialization: str
    run_kind: str
    association_debug_summary: Dict[str, Dict[str, float | int | str]]


# --------------------------------------------------------
# Utility functions
# --------------------------------------------------------

def finite_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def finite_mean(values: Any) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def finite_median(values: Any) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def safe_sum(values: Any) -> int:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return 0
    arr = arr[np.isfinite(arr.astype(float))]
    if arr.size == 0:
        return 0
    return int(np.sum(arr.astype(int)))


def percent_gain(final_value: float, initial_value: float) -> float:
    if not np.isfinite(final_value) or not np.isfinite(initial_value) or abs(initial_value) < 1e-30:
        return float("nan")
    return 100.0 * (final_value - initial_value) / abs(initial_value)


def mbit_per_joule(bit_per_joule: float) -> float:
    if not np.isfinite(bit_per_joule):
        return float("nan")
    return bit_per_joule / 1e6


def bps_to_mbps(bps: float) -> float:
    if not np.isfinite(bps):
        return float("nan")
    return bps / 1e6


def fmt_float(x: Any, digits: int = 4) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def fmt_sci(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}e}"


def first_finite(values: Any) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def last_finite(values: Any) -> float:
    arr = finite_array(values)
    if arr.size == 0:
        return float("nan")
    return float(arr[-1])


def get_array(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    value = data.get(key)
    if value is None:
        return np.array([])
    return np.asarray(value)


def get_mean_array(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(data.get(key, np.array([])), dtype=float).reshape(-1)


def normalized_percent_curve(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return x and normalized improvement (%) from a history array."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    base = arr[0]
    if not np.isfinite(base) or abs(base) < 1e-30:
        return np.arange(arr.size), np.full(arr.size, np.nan, dtype=float)
    y = 100.0 * (arr - base) / abs(base)
    if X_START_AT_ONE:
        x = np.arange(1, arr.size + 1)
    else:
        x = np.arange(arr.size)
    return x, y


def normalized_to_final_curve(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return x and objective values normalized by the last finite value.

    This is used for the power-allocation Dinkelbach trajectory. It is a
    convergence visualization, not a relative EE-gain metric.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    final = arr[-1]
    if not np.isfinite(final) or abs(final) < 1e-30:
        if X_START_AT_ONE:
            x = np.arange(1, arr.size + 1)
        else:
            x = np.arange(arr.size)
        return x, np.full(arr.size, np.nan, dtype=float)
    y = arr / final
    if X_START_AT_ONE:
        x = np.arange(1, arr.size + 1)
    else:
        x = np.arange(arr.size)
    return x, y


def parse_tag(tag: str) -> Tuple[str, str, str]:
    if "tauKhalf" in tag:
        pilot = "tauKhalf"
    else:
        pilot = "tauK"

    if "random" in tag:
        init = "random"
    elif "strongest" in tag or "singleton" in tag:
        init = "strongest"
    elif "fullcandidate" in tag or "full" in tag:
        init = "full"
    else:
        init = "unknown"

    if tag.startswith("final"):
        run_kind = "main"
    elif tag.startswith("sen") or tag.startswith("sens"):
        run_kind = "sensitivity"
    else:
        run_kind = "unknown"

    return pilot, init, run_kind


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def parse_association_debug_summary(run_dir: Path) -> Dict[str, Dict[str, float | int | str]]:
    """Parse association_debug_summary.txt if it exists beside merged_data.npz."""
    candidates = [
        run_dir / "association_debug_summary.txt",
        run_dir.parent / "association_debug_summary.txt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}

    out: Dict[str, Dict[str, float | int | str]] = {}
    current: Optional[str] = None

    def parse_value(value: str) -> float | int | str:
        value = value.strip()
        try:
            if re.fullmatch(r"[-+]?\d+", value):
                return int(value)
            return float(value)
        except Exception:
            return value

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Scenario:"):
                current = line.split(":", 1)[1].strip()
                out[current] = {}
                continue
            if current and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_").replace("/", "_per_")
                key = re.sub(r"[^a-z0-9_]+", "", key)
                out[current][key] = parse_value(value)

    return out


def load_runs() -> Dict[str, RunData]:
    assigned = {
        tag: PAPER_DATA_ROOT / folder / "merged_data.npz"
        for tag, folder in PAPER_RUN_FOLDERS.items()
    }
    missing = [path for path in assigned.values() if not path.is_file()]
    if missing:
        expected = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"Missing paper data files:\n{expected}")

    runs: Dict[str, RunData] = {}
    for tag, path in assigned.items():
        pilot, init, run_kind = parse_tag(tag)
        data = load_npz(path)
        run_dir = path.parent
        assoc_debug = parse_association_debug_summary(run_dir)
        runs[tag] = RunData(
            tag=tag,
            path=path,
            data=data,
            pilot=pilot,
            initialization=init,
            run_kind=run_kind,
            association_debug_summary=assoc_debug,
        )

    return runs


def make_output_dirs(output_tag: str) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_tag.strip()) or DEFAULT_OUTPUT_TAG
    root = OUTPUT_ROOT / f"convergence_figures_{timestamp}_{clean_tag}"
    dirs = {
        "root": root,
        "pdf": root / "pdf",
        "png": root / "png",
        "tables": root / "tables",
        "metadata": root / "metadata",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# --------------------------------------------------------
# Metric extraction
# --------------------------------------------------------

def association_initial_ee_bit(run: RunData, scenario: str) -> float:
    key = f"{scenario}_assoc_outer_mean_ee_history_mean_available"
    value = first_finite(get_array(run.data, key))
    if np.isfinite(value):
        return value
    key = f"{scenario}_assoc_outer_mean_ee_history_mean_padded"
    return first_finite(get_array(run.data, key))


def association_final_ee_bit(run: RunData, scenario: str) -> float:
    key = f"{scenario}_assoc_final_ee_bit_per_joule_per_setup_all"
    value = finite_mean(get_array(run.data, key))
    if np.isfinite(value):
        return value
    key = f"{scenario}_assoc_outer_mean_ee_history_mean_available"
    return last_finite(get_array(run.data, key))


def association_curve(run: RunData, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    if USE_PADDED_CURVES_FOR_PAPER:
        keys = [
            f"{scenario}_assoc_outer_mean_ee_history_mean_padded",
            f"{scenario}_assoc_outer_mean_ee_history_mean_available",
        ]
    else:
        keys = [
            f"{scenario}_assoc_outer_mean_ee_history_mean_available",
            f"{scenario}_assoc_outer_mean_ee_history_mean_padded",
        ]

    arr = np.array([])
    for key in keys:
        arr = get_mean_array(run.data, key)
        if finite_array(arr).size > 0:
            break
    return normalized_percent_curve(arr)


def power_curve(run: RunData, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    if USE_PADDED_CURVES_FOR_PAPER:
        keys = [
            f"{scenario}_power_alpha_padded_mean",
            f"{scenario}_power_alpha_available_mean",
        ]
    else:
        keys = [
            f"{scenario}_power_alpha_available_mean",
            f"{scenario}_power_alpha_padded_mean",
        ]

    arr = np.array([])
    for key in keys:
        arr = get_mean_array(run.data, key)
        if finite_array(arr).size > 0:
            break
    return normalized_to_final_curve(arr)


def power_initial_alpha_bit(run: RunData, scenario: str) -> float:
    key = f"{scenario}_power_alpha_available_mean"
    value = first_finite(get_array(run.data, key))
    if np.isfinite(value):
        return value
    key = f"{scenario}_power_alpha_padded_mean"
    return first_finite(get_array(run.data, key))


def power_final_alpha_bit(run: RunData, scenario: str) -> float:
    key = f"{scenario}_power_alpha_available_mean"
    value = last_finite(get_array(run.data, key))
    if np.isfinite(value):
        return value
    key = f"{scenario}_power_alpha_padded_mean"
    return last_finite(get_array(run.data, key))


def final_aps_per_user_for_power(run: RunData, scenario: str) -> float:
    """Return the final serving-set density used by power allocation.

    The power stage is run on the association-optimized sets, so the association
    final APs/user is a valid fallback when a power-specific key is unavailable.
    """
    for key in [
        f"{scenario}_power_final_avg_serving_aps_per_user_per_setup_all",
        f"{scenario}_assoc_final_avg_serving_aps_per_user_per_setup_all",
    ]:
        value = finite_mean(get_array(run.data, key))
        if np.isfinite(value):
            return value
    return float("nan")


def count_array_size(run: RunData, key: str) -> int:
    arr = get_array(run.data, key)
    if arr.size == 0:
        return 0
    return int(arr.reshape(-1).size)


def summarize_association(run: RunData, scenario: str) -> Dict[str, Any]:
    initial_bit = association_initial_ee_bit(run, scenario)
    final_bit = association_final_ee_bit(run, scenario)
    final_ee_arr = get_array(run.data, f"{scenario}_assoc_final_ee_bit_per_joule_per_setup_all")
    final_ap_arr = get_array(run.data, f"{scenario}_assoc_final_avg_serving_aps_per_user_per_setup_all")
    converged_arr = get_array(run.data, f"{scenario}_assoc_converged_all")
    outer_arr = get_array(run.data, f"{scenario}_assoc_num_outer_iterations_all")
    runtime_arr = get_array(run.data, f"{scenario}_assoc_runtime_per_setup_seconds_all")
    residual_arr = get_array(run.data, f"{scenario}_assoc_residual_history_mean_available")

    debug = run.association_debug_summary.get(scenario, {})

    return {
        "Tag": run.tag,
        "Pilot": PILOT_LABELS.get(run.pilot, run.pilot),
        "Initialization": INIT_LABELS.get(run.initialization, run.initialization),
        "Scenario": SCENARIO_LABELS.get(scenario, scenario),
        "Available setups": count_array_size(run, f"{scenario}_assoc_final_ee_bit_per_joule_per_setup_all"),
        "Chunks": count_array_size(run, f"{scenario}_assoc_converged_all"),
        "Converged chunks": safe_sum(converged_arr),
        "Initial EE (Mbit/J)": mbit_per_joule(initial_bit),
        "Final EE (Mbit/J)": mbit_per_joule(final_bit),
        "EE improvement (%)": percent_gain(final_bit, initial_bit),
        "Final APs/user": finite_mean(final_ap_arr),
        "Mean outer iterations": finite_mean(outer_arr),
        "Runtime/setup (s)": finite_mean(runtime_arr),
        "Final residual": last_finite(residual_arr),
        "Users with one action only": debug.get("users_with_one_action_only", "NA"),
        "Fraction users changed": debug.get("fraction_users_changed_initialtofinal_mean", debug.get("fraction_users_changed_mean", "NA")),
        "Cycle detected count": debug.get("cycle_detected_count", "NA"),
        "Terminal differs from returned setups": debug.get("terminal_differs_from_returned_setups", "NA"),
        "Best global EE gain over returned mean": debug.get("bestglobalee_gain_over_returned_mean", debug.get("best_global_ee_gain_over_returned_mean", "NA")),
    }


def summarize_power(run: RunData, scenario: str) -> Dict[str, Any]:
    initial_alpha = power_initial_alpha_bit(run, scenario)
    final_alpha = power_final_alpha_bit(run, scenario)

    final_ee_arr = get_array(run.data, f"{scenario}_power_final_ee_mbit_per_joule_per_setup_all")
    final_total_power = get_array(run.data, f"{scenario}_power_final_total_power_watt_per_setup_all")
    final_sum_rate = get_array(run.data, f"{scenario}_power_final_sum_rate_bps_per_setup_all")
    converged = get_array(run.data, f"{scenario}_power_converged_dinkelbach_all")
    rejected = get_array(run.data, f"{scenario}_power_stopped_on_rejected_candidate_all")
    true_hit = get_array(run.data, f"{scenario}_power_true_hit_max_dinkelbach_all")
    other_nonconv = get_array(run.data, f"{scenario}_power_other_nonconverged_all")
    qos = get_array(run.data, f"{scenario}_power_final_qos_feasible_all")
    safe = get_array(run.data, f"{scenario}_power_safe_point_found_all")
    users_below = get_array(run.data, f"{scenario}_power_final_num_users_below_safe_floor_all")
    min_qos = get_array(run.data, f"{scenario}_power_final_min_qos_margin_all")
    runtime = get_array(run.data, f"{scenario}_power_runtime_per_setup_seconds_all")
    num_iter = get_array(run.data, f"{scenario}_power_num_dinkelbach_iters_all")

    residual = get_array(run.data, f"{scenario}_power_residual_available_mean")
    if finite_array(residual).size == 0:
        residual = get_array(run.data, f"{scenario}_power_residual_padded_mean")

    return {
        "Tag": run.tag,
        "Pilot": PILOT_LABELS.get(run.pilot, run.pilot),
        "Initialization": INIT_LABELS.get(run.initialization, run.initialization),
        "Scenario": SCENARIO_LABELS.get(scenario, scenario),
        "Available setups": count_array_size(run, f"{scenario}_power_final_ee_mbit_per_joule_per_setup_all"),
        "Converged setups": safe_sum(converged),
        "Rejected-candidate stops": safe_sum(rejected),
        "True hit-max setups": safe_sum(true_hit),
        "Other nonconverged setups": safe_sum(other_nonconv),
        "QoS feasible setups": safe_sum(qos),
        "Safe point found setups": safe_sum(safe),
        "Initial alpha/EE (Mbit/J)": mbit_per_joule(initial_alpha),
        "Final alpha/EE (Mbit/J)": mbit_per_joule(final_alpha),
        "Alpha/EE improvement (%)": percent_gain(final_alpha, initial_alpha),
        "Final evaluated EE (Mbit/J)": finite_mean(final_ee_arr),
        "Final total power (W)": finite_mean(final_total_power),
        "Final sum rate (Mbps)": bps_to_mbps(finite_mean(final_sum_rate)),
        "Final APs/user": final_aps_per_user_for_power(run, scenario),
        "Mean users below floor": finite_mean(users_below),
        "Mean min QoS margin": finite_mean(min_qos),
        "Mean Dinkelbach iterations": finite_mean(num_iter),
        "Runtime/setup (s)": finite_mean(runtime),
        "Final residual": last_finite(residual),
    }


# --------------------------------------------------------
# Plotting
# --------------------------------------------------------

def save_figure(fig: plt.Figure, basename: str, dirs: Dict[str, Path]) -> None:
    pdf_path = dirs["pdf"] / f"{basename}.pdf"
    png_path = dirs["png"] / f"{basename}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=DPI)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.65)
    ax.tick_params(axis="both", which="major", length=2.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _compact_tick_label(value: float, _pos: int) -> str:
    """Format ticks as 0, 2.5, 5, ... instead of 0.0, 2.5, 5.0."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.94,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.9),
    )


def plot_run_panel(
    ax: plt.Axes,
    run: RunData,
    kind: str,
    panel_label: str,
    show_ylabel: bool = True,
) -> None:
    """Plot No RIS and Single RIS curves for one run and one optimizer kind."""
    line_styles = {
        "no_ris": "-",
        "single_ris": "--",
    }
    markers = {
        "no_ris": "o",
        "single_ris": "s",
    }

    for scenario in SCENARIOS:
        if kind == "association":
            x, y = association_curve(run, scenario)
        elif kind == "power":
            x, y = power_curve(run, scenario)
        else:
            raise ValueError(f"Unknown plot kind: {kind}")

        if x.size == 0:
            continue

        ax.plot(
            x,
            y,
            linestyle=line_styles[scenario],
            marker=markers[scenario],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markevery=max(1, len(x) // 5),
            label=SCENARIO_LABELS[scenario],
        )

    ax.set_xlabel("Outer Dinkelbach iteration")
    if show_ylabel:
        if kind == "association":
            ax.set_ylabel("Association EE improvement (%)")
        elif kind == "power":
            ax.set_ylabel(r"$\alpha^{(n)}/\alpha^{(\mathrm{final})}$")
        else:
            ax.set_ylabel("Normalized value")

    if kind == "association":
        ax.set_ylim(*ASSOCIATION_YLIM)
        ax.set_yticks(ASSOCIATION_YTICKS)
        ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))

    style_axis(ax)
    add_panel_label(ax, panel_label)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.82,
        borderpad=0.25,
        handlelength=1.5,
        labelspacing=0.25,
        borderaxespad=0.25,
    )


def generate_convergence_figures(kind: str, runs: Dict[str, RunData], dirs: Dict[str, Path]) -> None:
    """Generate side-by-side, top-bottom, and standalone convergence figures."""
    required = [MAIN_RANDOM_TAG_TAUK, MAIN_RANDOM_TAG_TAUKHALF]
    missing = [tag for tag in required if tag not in runs]
    if missing:
        print(f"WARNING: cannot generate {kind} main figures. Missing: {missing}")
        return

    tauK_run = runs[MAIN_RANDOM_TAG_TAUK]
    tauKhalf_run = runs[MAIN_RANDOM_TAG_TAUKHALF]

    if kind == "association":
        prefix = "fig1_association_normalized"
    elif kind == "power":
        prefix = "fig2_power_dinkelbach_normalized"
    else:
        raise ValueError(kind)

    # Style A: side-by-side
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SIDE_BY_SIDE, sharey=False)
    plot_run_panel(axes[0], tauK_run, kind, PILOT_LABELS["tauK"], show_ylabel=True)
    plot_run_panel(axes[1], tauKhalf_run, kind, PILOT_LABELS["tauKhalf"], show_ylabel=False)
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, f"{prefix}_side_by_side", dirs)

    # Style C: top-bottom
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TOP_BOTTOM, sharex=False)
    plot_run_panel(axes[0], tauK_run, kind, PILOT_LABELS["tauK"], show_ylabel=True)
    plot_run_panel(axes[1], tauKhalf_run, kind, PILOT_LABELS["tauKhalf"], show_ylabel=True)
    fig.tight_layout(h_pad=0.7)
    save_figure(fig, f"{prefix}_top_bottom", dirs)

    # Standalone tauK
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_STANDALONE)
    plot_run_panel(ax, tauK_run, kind, PILOT_LABELS["tauK"], show_ylabel=True)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_tauK_standalone", dirs)

    # Standalone tauKhalf
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_STANDALONE)
    plot_run_panel(ax, tauKhalf_run, kind, PILOT_LABELS["tauKhalf"], show_ylabel=True)
    fig.tight_layout()
    save_figure(fig, f"{prefix}_tauKhalf_standalone", dirs)


# --------------------------------------------------------
# Table writing
# --------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stringify_value(value: Any) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "NA"
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def format_readable_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "No rows."

    string_rows: List[Dict[str, str]] = []
    widths = {col: len(col) for col in columns}
    for row in rows:
        sr = {col: stringify_value(row.get(col, "")) for col in columns}
        string_rows.append(sr)
        for col in columns:
            widths[col] = max(widths[col], len(sr[col]))

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    lines = [header, sep]
    for sr in string_rows:
        lines.append(" | ".join(sr[col].ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def write_readable_report(
    path: Path,
    runs: Dict[str, RunData],
    association_rows: List[Dict[str, Any]],
    power_rows: List[Dict[str, Any]],
    paper_association_rows: List[Dict[str, Any]],
    paper_power_rows: List[Dict[str, Any]],
) -> None:
    assoc_cols = [
        "Pilot",
        "Initialization",
        "Scenario",
        "Available setups",
        "Chunks",
        "Converged chunks",
        "Initial EE (Mbit/J)",
        "Final EE (Mbit/J)",
        "EE improvement (%)",
        "Final APs/user",
        "Mean outer iterations",
        "Runtime/setup (s)",
        "Users with one action only",
        "Fraction users changed",
    ]

    power_cols = [
        "Pilot",
        "Initialization",
        "Scenario",
        "Available setups",
        "Converged setups",
        "Rejected-candidate stops",
        "True hit-max setups",
        "QoS feasible setups",
        "Safe point found setups",
        "Final evaluated EE (Mbit/J)",
        "Final total power (W)",
        "Final sum rate (Mbps)",
        "Final APs/user",
        "Mean Dinkelbach iterations",
        "Runtime/setup (s)",
        "Initial alpha/EE (Mbit/J)",
        "Final alpha/EE (Mbit/J)",
        "Alpha/EE improvement (%)",
    ]

    paper_assoc_cols = [
        "Pilot",
        "Initialization",
        "Scenario",
        "Initial EE (Mbit/J)",
        "Final EE (Mbit/J)",
        "Gain (%)",
        "APs/user",
        "Outer iter.",
        "Runtime (s)",
        "Conv. chunks",
    ]

    paper_power_cols = [
        "Pilot",
        "Initialization",
        "Scenario",
        "EE (Mbit/J)",
        "Rate (Mbps)",
        "Power (W)",
        "APs/user",
        "Conv.",
        "Reject",
        "QoS",
        "Safe",
        "Iter.",
        "Runtime (s)",
    ]

    lines: List[str] = []
    lines.append("RIS-aided cell-free convergence processing report")
    lines.append("=" * 80)
    lines.append(f"Created at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("Selected NPZ files")
    lines.append("-" * 80)
    for tag, run in sorted(runs.items()):
        lines.append(f"{tag}: {run.path}")
    lines.append("")

    lines.append("Data-quality notes")
    lines.append("-" * 80)
    for tag, run in sorted(runs.items()):
        for scenario in SCENARIOS:
            assoc_n = count_array_size(run, f"{scenario}_assoc_final_ee_bit_per_joule_per_setup_all")
            power_n = count_array_size(run, f"{scenario}_power_final_ee_mbit_per_joule_per_setup_all")
            lines.append(
                f"{tag} | {SCENARIO_LABELS[scenario]}: association setups={assoc_n}, power setups={power_n}"
            )
    lines.append("")

    lines.append("Table 1. Association game initialization sensitivity")
    lines.append("=" * 80)
    lines.append(format_readable_table(association_rows, assoc_cols))
    lines.append("")

    lines.append("Table 2. Power optimizer behavior")
    lines.append("=" * 80)
    lines.append(format_readable_table(power_rows, power_cols))
    lines.append("")

    lines.append("Paper-ready compact association table")
    lines.append("=" * 80)
    lines.append(format_readable_table(paper_association_rows, paper_assoc_cols))
    lines.append("")

    lines.append("Paper-ready compact power table")
    lines.append("=" * 80)
    lines.append(format_readable_table(paper_power_rows, paper_power_cols))
    lines.append("")

    lines.append("Interpretation reminders")
    lines.append("-" * 80)
    lines.append("1. Association figures use 100*(EE_n - EE_1)/abs(EE_1), which is a relative EE-improvement metric.")
    lines.append("2. Power figures use alpha_n/alpha_final, which is a convergence visualization of the Dinkelbach trajectory.")
    lines.append("3. The alpha-improvement percentage is retained only as a diagnostic and should not be used as the headline power result.")
    lines.append("4. Final evaluated EE is computed from the final returned power vector and should be used for final system-performance claims.")
    lines.append("5. Rejected-candidate stops mean that the power stage returned the last accepted feasible incumbent.")
    lines.append("6. Missing setups are ignored automatically by NumPy arrays already present in each merged NPZ.")
    lines.append("7. For the final paper, report the available setup count when it is smaller than the nominal run size.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_tables(runs: Dict[str, RunData]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    association_rows: List[Dict[str, Any]] = []
    power_rows: List[Dict[str, Any]] = []

    for tag in KNOWN_TAGS:
        if tag not in runs:
            continue
        run = runs[tag]
        for scenario in SCENARIOS:
            association_rows.append(summarize_association(run, scenario))
            power_rows.append(summarize_power(run, scenario))

    # Include any extra selected runs after the known ones.
    for tag, run in sorted(runs.items()):
        if tag in KNOWN_TAGS:
            continue
        for scenario in SCENARIOS:
            association_rows.append(summarize_association(run, scenario))
            power_rows.append(summarize_power(run, scenario))

    return association_rows, power_rows


def ratio_string(numerator: Any, denominator: Any) -> str:
    try:
        num = int(numerator)
        den = int(denominator)
    except Exception:
        return "NA"
    if den <= 0:
        return "NA"
    return f"{num}/{den}"


def build_paper_power_rows(power_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a compact table suitable for copying into the paper."""
    rows: List[Dict[str, Any]] = []
    for row in power_rows:
        available = row.get("Available setups", 0)
        rows.append({
            "Pilot": row.get("Pilot", ""),
            "Initialization": row.get("Initialization", ""),
            "Scenario": row.get("Scenario", ""),
            "EE (Mbit/J)": row.get("Final evaluated EE (Mbit/J)", float("nan")),
            "Rate (Mbps)": row.get("Final sum rate (Mbps)", float("nan")),
            "Power (W)": row.get("Final total power (W)", float("nan")),
            "APs/user": row.get("Final APs/user", float("nan")),
            "Conv.": ratio_string(row.get("Converged setups", 0), available),
            "Reject": ratio_string(row.get("Rejected-candidate stops", 0), available),
            "QoS": ratio_string(row.get("QoS feasible setups", 0), available),
            "Safe": ratio_string(row.get("Safe point found setups", 0), available),
            "Iter.": row.get("Mean Dinkelbach iterations", float("nan")),
            "Runtime (s)": row.get("Runtime/setup (s)", float("nan")),
        })
    return rows


def build_paper_association_rows(association_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a compact association table suitable for copying into the paper."""
    rows: List[Dict[str, Any]] = []
    for row in association_rows:
        rows.append({
            "Pilot": row.get("Pilot", ""),
            "Initialization": row.get("Initialization", ""),
            "Scenario": row.get("Scenario", ""),
            "Initial EE (Mbit/J)": row.get("Initial EE (Mbit/J)", float("nan")),
            "Final EE (Mbit/J)": row.get("Final EE (Mbit/J)", float("nan")),
            "Gain (%)": row.get("EE improvement (%)", float("nan")),
            "APs/user": row.get("Final APs/user", float("nan")),
            "Outer iter.": row.get("Mean outer iterations", float("nan")),
            "Runtime (s)": row.get("Runtime/setup (s)", float("nan")),
            "Conv. chunks": ratio_string(row.get("Converged chunks", 0), row.get("Chunks", 0)),
        })
    return rows


# --------------------------------------------------------
# Metadata writing
# --------------------------------------------------------

def write_metadata(dirs: Dict[str, Path], runs: Dict[str, RunData]) -> None:
    selected = {
        tag: {
            "path": str(run.path),
            "pilot": run.pilot,
            "initialization": run.initialization,
            "run_kind": run.run_kind,
            "num_keys": len(run.data),
        }
        for tag, run in sorted(runs.items())
    }
    with open(dirs["metadata"] / "selected_files.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    key_report_lines = []
    for tag, run in sorted(runs.items()):
        key_report_lines.append(f"{tag}")
        key_report_lines.append("-" * 80)
        for key in sorted(run.data.keys()):
            arr = np.asarray(run.data[key])
            key_report_lines.append(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        key_report_lines.append("")
    with open(dirs["metadata"] / "npz_key_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(key_report_lines))


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main() -> None:
    configure_matplotlib()

    runs = load_runs()

    print("\nDetected runs:")
    for tag, run in sorted(runs.items()):
        print(f"  {tag}: {run.path}")

    dirs = make_output_dirs(DEFAULT_OUTPUT_TAG)

    # Build tables before plots so data issues are visible early.
    association_rows, power_rows = build_tables(runs)
    paper_association_rows = build_paper_association_rows(association_rows)
    paper_power_rows = build_paper_power_rows(power_rows)

    write_csv(dirs["tables"] / "association_initialization_sensitivity.csv", association_rows)
    write_csv(dirs["tables"] / "power_optimizer_behavior.csv", power_rows)
    write_csv(dirs["tables"] / "paper_association_table.csv", paper_association_rows)
    write_csv(dirs["tables"] / "paper_power_table.csv", paper_power_rows)
    write_readable_report(
        dirs["tables"] / "table_values_readable.txt",
        runs,
        association_rows,
        power_rows,
        paper_association_rows,
        paper_power_rows,
    )
    write_metadata(dirs, runs)

    generate_convergence_figures("association", runs, dirs)
    generate_convergence_figures("power", runs, dirs)

    print("\nDone.")
    print(f"Output folder: {dirs['root'].resolve()}")
    print("\nImportant files:")
    print(f"  Readable table values: {dirs['tables'] / 'table_values_readable.txt'}")
    print(f"  Association CSV:       {dirs['tables'] / 'association_initialization_sensitivity.csv'}")
    print(f"  Power CSV:             {dirs['tables'] / 'power_optimizer_behavior.csv'}")
    print(f"  Paper association CSV: {dirs['tables'] / 'paper_association_table.csv'}")
    print(f"  Paper power CSV:       {dirs['tables'] / 'paper_power_table.csv'}")
    print(f"  PDF figures:           {dirs['pdf']}")
    print(f"  PNG figures:           {dirs['png']}")


if __name__ == "__main__":
    main()
