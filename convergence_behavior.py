# ITNG YMYFA 80asra YA
# IMZZ


# convergence_behavior.py

"""
convergence_behavior.py

Generate convergence-behavior data for two RIS-aided cell-free optimizers:

1) Association game optimizer
   - Runs with equal-power redistribution, as designed in association_game_optimizer.py.
   - Intended main plots:
       a) mean EE vs outer Dinkelbach iteration
       b) relative Dinkelbach residual vs outer Dinkelbach iteration

2) Optimized downlink power allocation
   - Runs on the association-optimized serving sets returned by the association game.
   - Intended main plots:
       a) Dinkelbach alpha, equivalent to achieved EE, vs outer iteration
       b) relative Dinkelbach residual vs outer iteration

Output layout:
    results/convergence/convergence_YYYYMMDD_HHMMSS_tag/
        merged_metadata.json
        merged_summary.txt
        merged_data.npz
        chunks/
            metadata_chunk_000.json
            summary_chunk_000.txt
            data_chunk_000.npz
            ...

The script supports both desktop execution and Slurm array execution. It parallelizes
only across chunks, never inside Dinkelbach or SCA iterations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

# Project imports. This script is intended to be placed in the project root,
# next to the src/ package.
from src.config import Config, make_default_config
from src.simulation_runner import apply_config_overrides
from src.geometry import generate_geometry_setups
from src.large_scale import compute_large_scale_setups
from src.correlation_setups import build_correlation_setups
from src.ris_phase_shifts import build_ris_phase_shifts_setups
from src.pilot_assignment import assign_pilots_setups
from src.channel_estimation import build_channel_estimation_statistics
from src.power_allocation import (
    allocate_downlink_power,
    OptimizedPowerOptions,
)
from src.association_game_optimizer import (
    GameOptions,
    optimize_association_game_ee,
)
from src.dl_sinr_closed_form import compute_dl_sinr_closed_form
from src.spectral_efficiency import compute_spectral_efficiency_setups
from src.energy_efficiency import compute_energy_efficiency_setups


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

SCENARIOS = ("no_ris", "single_ris")


# -----------------------------------------------------------------------------
# Experiment profile
# -----------------------------------------------------------------------------
# Change the main simulation/default experiment values here.
# Slurm should mainly control resources and array size. The Python file controls
# system dimensions, pilot length, seeds, and optimizer defaults.

EXPERIMENT_TOTAL_SETUPS = 100
EXPERIMENT_DEFAULT_TOTAL_CHUNKS = 20
EXPERIMENT_BASE_SEED = 20260507

EXPERIMENT_CONFIG_OVERRIDES: dict[str, object] = {
    # System dimensions
    "dims.num_users": 10,
    "dims.num_aps": 35,
    "dims.num_ap_antennas": 4,

    # RIS size. The scenario function controls num_ris = 0 or 1.
    "dims.ris_n_hor": 10,
    "dims.ris_n_ver": 10,

    # Orthogonal pilots: None means tau_p = K.
    # For pilot contamination with K=10, change this to 5.
    # For K=14, change this to 7.
    "pilots.pilot_len": 5,
    "pilots.coherence_block_length": 200.0,
    "pilots.pilot_power_watt": 0.1,

    # Bandwidth and powers
    "noise.bandwidth_hz": 1e6,
    "power.ul_max_power_watt_per_user": 0.2,
    "power.dl_max_power_watt_per_ap": 1.0,

    # Geometry
    "geom.wraparound_enabled": False,
    "geom.ap_placement": "random",
    "geom.ris_placement": "edge",
    "geom.ris_edge_angle_deg": 180.0,
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_slurm_environment() -> bool:
    return "SLURM_JOB_ID" in os.environ or "SLURM_ARRAY_TASK_ID" in os.environ


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonable(obj: Any) -> Any:
    """Convert common Python, NumPy, dataclass, and Path objects to JSON-safe objects."""
    if is_dataclass(obj):
        return jsonable(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def parse_json_or_path(value: Optional[str]) -> dict[str, Any]:
    if value is None or value.strip() == "":
        return {}
    candidate = Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


def compute_chunk_range(total_setups: int, total_chunks: int, chunk_id: int) -> tuple[int, int, int]:
    if total_setups <= 0:
        raise ValueError("total_setups must be positive")
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if chunk_id < 0 or chunk_id >= total_chunks:
        raise ValueError(f"chunk_id={chunk_id} is outside [0, {total_chunks - 1}]")

    base = total_setups // total_chunks
    rem = total_setups % total_chunks
    start = chunk_id * base + min(chunk_id, rem)
    n = base + (1 if chunk_id < rem else 0)
    end = start + n
    return start, end, n


def stage_seeds(base_seed: int) -> dict[str, int]:
    """Keep the same separation style used by simulation_runner.py."""
    return {
        "geometry": base_seed + 101,
        "large_scale": base_seed + 202,
        "phase": base_seed + 303,
        "pilot": base_seed + 404,
        "channels": base_seed + 505,
        "estimation": base_seed + 606,
    }


def safe_last(seq: Iterable[Any], default: float = np.nan) -> float:
    vals = list(seq)
    if not vals:
        return float(default)
    try:
        return float(vals[-1])
    except Exception:
        return float(default)


def safe_first(seq: Iterable[Any], default: float = np.nan) -> float:
    vals = list(seq)
    if not vals:
        return float(default)
    try:
        return float(vals[0])
    except Exception:
        return float(default)


def finite_mean(x: Any) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def finite_median(x: Any) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def rel_improvement(final: float, initial: float) -> float:
    if not np.isfinite(final) or not np.isfinite(initial) or abs(initial) <= 1e-30:
        return float("nan")
    return float((final - initial) / abs(initial))


def pad_1d(values: Iterable[float], width: int, *, mode: str = "last") -> np.ndarray:
    """
    Pad a 1D history to a given width.

    mode='last' repeats the final available value.
    mode='nan' pads with NaN.
    mode='zero' pads with 0.
    """
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    out = np.full(width, np.nan, dtype=float)
    n = min(arr.size, width)
    if n > 0:
        out[:n] = arr[:n]
    if n < width:
        if mode == "last":
            if n > 0:
                out[n:] = out[n - 1]
            # If n == 0, keep NaN padding.
        elif mode == "zero":
            out[n:] = 0.0
        elif mode == "nan":
            pass
        else:
            raise ValueError(f"Unknown pad mode {mode!r}")
    return out


def pad_histories(histories: list[Iterable[float]], *, mode: str = "last") -> np.ndarray:
    width = max((len(list(h)) for h in histories), default=0)
    if width == 0:
        return np.empty((len(histories), 0), dtype=float)
    # list(h) above consumes only if h is iterator. Our inputs are lists, but be safe.
    hist_lists = [list(h) for h in histories]
    width = max((len(h) for h in hist_lists), default=0)
    return np.vstack([pad_1d(h, width, mode=mode) for h in hist_lists])


def pad_matrix_columns(mat: np.ndarray, width: int, *, mode: str = "last") -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat[None, :]
    rows, cols = mat.shape
    if cols == width:
        return mat.copy()
    if cols > width:
        return mat[:, :width].copy()
    out = np.full((rows, width), np.nan, dtype=float)
    if cols > 0:
        out[:, :cols] = mat
    if cols < width:
        if mode == "last" and cols > 0:
            out[:, cols:] = mat[:, cols - 1][:, None]
        elif mode == "zero":
            out[:, cols:] = 0.0
        elif mode == "nan":
            pass
        else:
            raise ValueError(f"Unknown pad mode {mode!r}")
    return out


def weighted_nanmean_rows(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean over rows with NaN ignored. Returns mean and available count."""
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if matrix.shape[0] != weights.size:
        raise ValueError("matrix row count must match weights")
    valid = np.isfinite(matrix)
    weighted = np.where(valid, matrix * weights[:, None], 0.0)
    denom = np.sum(np.where(valid, weights[:, None], 0.0), axis=0)
    out = np.full(matrix.shape[1], np.nan, dtype=float)
    np.divide(np.sum(weighted, axis=0), denom, out=out, where=denom > 0)
    count = np.sum(valid, axis=0).astype(int)
    return out, count


def npz_get(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    return data[key] if key in data.files else default


# -----------------------------------------------------------------------------
# Association debug helpers
# -----------------------------------------------------------------------------


def _mask_per_user_sets(mask: np.ndarray) -> list[list[int]]:
    """
    Convert a serving mask into readable AP sets per user.

    Accepts shape (L, K) or (1, L, K).
    Returns list of length K.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m[0]
    L, K = m.shape
    return [np.where(m[:, k])[0].astype(int).tolist() for k in range(K)]


def _mask_serving_counts(mask: np.ndarray) -> np.ndarray:
    """
    Return number of serving APs per user.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3:
        m = m[0]
    return np.sum(m, axis=0).astype(int)


def _pad_list(values: list[float], width: int) -> np.ndarray:
    out = np.full(width, np.nan, dtype=float)
    n = min(len(values), width)
    if n > 0:
        out[:n] = np.asarray(values[:n], dtype=float)
    return out


def build_association_debug_arrays(prefix: str, assoc: Any) -> dict[str, np.ndarray]:
    """
    Build per-setup debug arrays for the association game.

    These are not used by the optimizer. They are for diagnosing:
    - action-space size,
    - initial vs final serving sets,
    - utility trajectory,
    - global EE trajectory,
    - best utility mask vs best global EE mask.
    """
    final_mask = np.asarray(assoc.final_serve_mask, dtype=bool)  # (S, L, K)
    S, L, K = final_mask.shape

    first_outer = assoc.fixed_alpha_results_by_outer_iter[0]
    last_outer = assoc.fixed_alpha_results_by_outer_iter[-1]

    initial_mask = np.concatenate([r.initial_serve_mask for r in first_outer], axis=0)
    terminal_mask = np.concatenate([r.terminal_serve_mask for r in last_outer], axis=0)
    best_utility_mask = np.concatenate([r.best_utility_serve_mask for r in last_outer], axis=0)
    best_global_ee_mask = np.concatenate([r.best_global_ee_serve_mask for r in last_outer], axis=0)

    candidate_sizes = np.zeros((S, K), dtype=int)
    action_counts = np.zeros((S, K), dtype=int)

    for s, graph in enumerate(assoc.graphs):
        for k in range(K):
            candidate_sizes[s, k] = int(len(graph.candidate_aps_per_user[k]))
            action_counts[s, k] = int(len(graph.action_masks_per_user[k]))

    initial_counts = np.sum(initial_mask, axis=1).astype(int)
    final_counts = np.sum(final_mask, axis=1).astype(int)
    terminal_counts = np.sum(terminal_mask, axis=1).astype(int)
    best_utility_counts = np.sum(best_utility_mask, axis=1).astype(int)
    best_global_ee_counts = np.sum(best_global_ee_mask, axis=1).astype(int)

    users_changed = np.any(initial_mask != final_mask, axis=1).astype(int)
    hamming_distance = np.sum(initial_mask != final_mask, axis=(1, 2)).astype(int)

    terminal_differs_from_returned = np.any(terminal_mask != final_mask, axis=(1, 2)).astype(int)
    best_global_ee_differs_from_returned = np.any(best_global_ee_mask != final_mask, axis=(1, 2)).astype(int)

    returned_ee = np.asarray(assoc.final_ee.ee_bit_per_joule, dtype=float)
    best_global_ee = np.asarray([float(r.best_global_ee_bit_per_joule) for r in last_outer], dtype=float)
    best_global_ee_gain_over_returned = np.full(S, np.nan, dtype=float)
    denom = np.maximum(np.abs(returned_ee), 1e-12)
    best_global_ee_gain_over_returned = (best_global_ee - returned_ee) / denom

    max_outer = len(assoc.fixed_alpha_results_by_outer_iter)
    max_inner = 0
    for outer in assoc.fixed_alpha_results_by_outer_iter:
        for r in outer:
            max_inner = max(max_inner, len(r.utility_proxy_history))
            max_inner = max(max_inner, len(r.ee_bit_per_joule_history))

    utility_hist = np.full((max_outer, S, max_inner), np.nan, dtype=float)
    ee_hist = np.full((max_outer, S, max_inner), np.nan, dtype=float)
    changed_hist = np.full((max_outer, S, max_inner), np.nan, dtype=float)
    cycle_detected = np.zeros((max_outer, S), dtype=int)
    converged = np.zeros((max_outer, S), dtype=int)
    inner_iterations = np.zeros((max_outer, S), dtype=int)

    for o, outer in enumerate(assoc.fixed_alpha_results_by_outer_iter):
        for s, r in enumerate(outer):
            utility_hist[o, s, :] = _pad_list(r.utility_proxy_history, max_inner)
            ee_hist[o, s, :] = _pad_list(r.ee_bit_per_joule_history, max_inner)

            # num_users_changed_history has length equal to number of game updates.
            # Put it from index 1 onward because index 0 is the initial profile.
            ch = np.full(max_inner, np.nan, dtype=float)
            n = min(len(r.num_users_changed_history), max_inner - 1)
            if n > 0:
                ch[1:n + 1] = np.asarray(r.num_users_changed_history[:n], dtype=float)
            changed_hist[o, s, :] = ch

            cycle_detected[o, s] = int(bool(r.cycle_detected))
            converged[o, s] = int(bool(r.converged))
            inner_iterations[o, s] = int(r.num_iterations)

    return {
        f"{prefix}_debug_initial_serve_mask": initial_mask.astype(np.uint8),
        f"{prefix}_debug_final_serve_mask": final_mask.astype(np.uint8),
        f"{prefix}_debug_terminal_serve_mask": terminal_mask.astype(np.uint8),
        f"{prefix}_debug_best_utility_serve_mask": best_utility_mask.astype(np.uint8),
        f"{prefix}_debug_best_global_ee_serve_mask": best_global_ee_mask.astype(np.uint8),

        f"{prefix}_debug_candidate_sizes": candidate_sizes,
        f"{prefix}_debug_action_counts": action_counts,
        f"{prefix}_debug_initial_serving_counts": initial_counts,
        f"{prefix}_debug_final_serving_counts": final_counts,
        f"{prefix}_debug_terminal_serving_counts": terminal_counts,
        f"{prefix}_debug_best_utility_serving_counts": best_utility_counts,
        f"{prefix}_debug_best_global_ee_serving_counts": best_global_ee_counts,

        f"{prefix}_debug_users_changed_initial_final": users_changed,
        f"{prefix}_debug_fraction_users_changed_per_setup": np.mean(users_changed, axis=1),
        f"{prefix}_debug_hamming_initial_final_per_setup": hamming_distance,
        f"{prefix}_debug_terminal_differs_from_returned": terminal_differs_from_returned,
        f"{prefix}_debug_best_global_ee_differs_from_returned": best_global_ee_differs_from_returned,

        f"{prefix}_debug_returned_ee_bit_per_joule": returned_ee,
        f"{prefix}_debug_best_global_ee_bit_per_joule": best_global_ee,
        f"{prefix}_debug_best_global_ee_gain_over_returned": best_global_ee_gain_over_returned,

        f"{prefix}_debug_inner_utility_proxy_history": utility_hist,
        f"{prefix}_debug_inner_ee_bit_per_joule_history": ee_hist,
        f"{prefix}_debug_inner_num_users_changed_history": changed_hist,
        f"{prefix}_debug_inner_cycle_detected": cycle_detected,
        f"{prefix}_debug_inner_converged": converged,
        f"{prefix}_debug_inner_iterations": inner_iterations,
    }


def build_association_debug_text(prefix: str, assoc: Any, arrays: dict[str, np.ndarray]) -> str:
    """
    Human-readable association debug report for one scenario in one chunk.
    """
    candidate_sizes = arrays[f"{prefix}_debug_candidate_sizes"]
    action_counts = arrays[f"{prefix}_debug_action_counts"]
    initial_counts = arrays[f"{prefix}_debug_initial_serving_counts"]
    final_counts = arrays[f"{prefix}_debug_final_serving_counts"]
    frac_changed = arrays[f"{prefix}_debug_fraction_users_changed_per_setup"]
    hamming = arrays[f"{prefix}_debug_hamming_initial_final_per_setup"]
    terminal_differs = arrays[f"{prefix}_debug_terminal_differs_from_returned"]
    best_ee_differs = arrays[f"{prefix}_debug_best_global_ee_differs_from_returned"]
    best_ee_gain = arrays[f"{prefix}_debug_best_global_ee_gain_over_returned"]
    returned_ee = arrays[f"{prefix}_debug_returned_ee_bit_per_joule"]
    best_ee = arrays[f"{prefix}_debug_best_global_ee_bit_per_joule"]

    initial_mask = arrays[f"{prefix}_debug_initial_serve_mask"].astype(bool)
    final_mask = arrays[f"{prefix}_debug_final_serve_mask"].astype(bool)

    lines: list[str] = []
    lines.append(f"Association debug report: {prefix}")
    lines.append("=" * 80)

    lines.append("Aggregate action-space diagnostics")
    lines.append("-" * 80)
    lines.append(f"setups: {candidate_sizes.shape[0]}")
    lines.append(f"users per setup: {candidate_sizes.shape[1]}")
    lines.append(f"candidate APs/user mean: {float(np.mean(candidate_sizes)):.4f}")
    lines.append(f"candidate APs/user min/max: {int(np.min(candidate_sizes))} / {int(np.max(candidate_sizes))}")
    lines.append(f"actions/user mean: {float(np.mean(action_counts)):.4f}")
    lines.append(f"actions/user min/max: {int(np.min(action_counts))} / {int(np.max(action_counts))}")
    lines.append(f"users with one action only: {int(np.sum(action_counts <= 1))}")
    lines.append("")

    lines.append("Initial vs final association")
    lines.append("-" * 80)
    lines.append(f"initial serving APs/user mean: {float(np.mean(initial_counts)):.4f}")
    lines.append(f"final serving APs/user mean: {float(np.mean(final_counts)):.4f}")
    lines.append(f"fraction users changed mean: {float(np.mean(frac_changed)):.4f}")
    lines.append(f"hamming distance/setup mean: {float(np.mean(hamming)):.4f}")
    lines.append("")

    lines.append("Incumbent and global-EE diagnostic")
    lines.append("-" * 80)
    lines.append(f"terminal mask differs from returned mask setups: {int(np.sum(terminal_differs))}")
    lines.append(f"best-global-EE mask differs from returned mask setups: {int(np.sum(best_ee_differs))}")
    lines.append(f"returned EE mean bit/J: {float(np.mean(returned_ee)):.6e}")
    lines.append(f"best global EE seen mean bit/J: {float(np.mean(best_ee)):.6e}")
    lines.append(f"best global EE gain over returned mean: {float(np.mean(best_ee_gain)):.6e}")
    lines.append("")

    lines.append("Per-setup compact details")
    lines.append("-" * 80)
    for s in range(candidate_sizes.shape[0]):
        lines.append(
            f"setup {s}: "
            f"cand_mean={float(np.mean(candidate_sizes[s])):.2f}, "
            f"act_mean={float(np.mean(action_counts[s])):.2f}, "
            f"init_AP/user={float(np.mean(initial_counts[s])):.2f}, "
            f"final_AP/user={float(np.mean(final_counts[s])):.2f}, "
            f"frac_changed={float(frac_changed[s]):.2f}, "
            f"hamming={int(hamming[s])}, "
            f"returned_EE={float(returned_ee[s]):.6e}, "
            f"best_global_EE={float(best_ee[s]):.6e}"
        )

        # Print user sets only for small chunks to keep reports readable.
        if candidate_sizes.shape[0] <= 2:
            init_sets = _mask_per_user_sets(initial_mask[s])
            final_sets = _mask_per_user_sets(final_mask[s])
            for k, (a, b) in enumerate(zip(init_sets, final_sets)):
                changed = "changed" if a != b else "same"
                lines.append(f"  user {k}: init={a} final={b} {changed}")

    lines.append("")
    return "\n".join(lines)


def write_association_debug_files(
    *,
    run_dir: Path,
    chunk_id: int,
    scenario: str,
    assoc: Any,
) -> None:
    """
    Write per-chunk association debug files.

    Output:
        chunks/association_debug/no_ris_association_debug_chunk_000.npz
        chunks/association_debug/no_ris_association_debug_chunk_000.txt
    """
    debug_dir = run_dir / "chunks" / "association_debug"
    ensure_dir(debug_dir)

    arrays = build_association_debug_arrays(scenario, assoc)

    npz_path = debug_dir / f"{scenario}_association_debug_chunk_{chunk_id:03d}.npz"
    txt_path = debug_dir / f"{scenario}_association_debug_chunk_{chunk_id:03d}.txt"

    np.savez_compressed(npz_path, **arrays)
    write_text(txt_path, build_association_debug_text(scenario, assoc, arrays) + "\n")



# -----------------------------------------------------------------------------
# Config and options
# -----------------------------------------------------------------------------


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    stamp = args.run_stamp or now_stamp()
    return (Path("results") / "convergence" / f"convergence_{stamp}_{args.tag}").resolve()


def scenario_overrides(scenario: str) -> dict[str, Any]:
    if scenario == "no_ris":
        return {
            "ris.enable_ris": False,
            "dims.num_ris": 0,
        }
    if scenario == "single_ris":
        return {
            "ris.enable_ris": True,
            "dims.num_ris": 1,
        }
    raise ValueError(f"Unknown scenario {scenario!r}")


def build_config_for_scenario(
    *,
    args: argparse.Namespace,
    scenario: str,
    chunk_seed: int,
    chunk_num_setups: int,
) -> Config:
    # Main defaults live in EXPERIMENT_CONFIG_OVERRIDES above.
    # Optional --config-overrides-json still works and has priority, but is no
    # longer required for normal runs.
    common_overrides = dict(EXPERIMENT_CONFIG_OVERRIDES)
    common_overrides.update(parse_json_or_path(args.config_overrides_json))

    overrides: dict[str, Any] = {}
    overrides.update(common_overrides)
    overrides.update(scenario_overrides(scenario))
    overrides["sim.num_setups"] = int(chunk_num_setups)
    overrides["sim.seed"] = int(chunk_seed)
    cfg = apply_config_overrides(make_default_config(), overrides)
    cfg.validate()
    return cfg


def build_game_options(args: argparse.Namespace, random_seed: int) -> GameOptions:
    return GameOptions(
        d_th=float(args.assoc_d_th),
        redistribution_scheme="equal",
        init_action_mode=args.assoc_init_action_mode,
        response_mode=args.assoc_response_mode,
        omega_initial=float(args.assoc_omega_initial),
        max_game_iters=int(args.assoc_max_game_iters),
        max_dinkelbach_iters=int(args.assoc_max_dinkelbach_iters),
        dinkelbach_tol=float(args.assoc_dinkelbach_tol),
        convergence_window=int(args.assoc_convergence_window),
        stability_tol=float(args.assoc_stability_tol),
        utility_improvement_tol=float(args.assoc_utility_improvement_tol),
        max_subset_size=args.assoc_max_subset_size,
        max_actions_per_user=args.assoc_max_actions_per_user,
        incumbent_tracking=bool(args.assoc_incumbent_tracking),
        outer_warm_start=bool(args.assoc_outer_warm_start),
        cycle_detection=bool(args.assoc_cycle_detection),
        random_seed=int(random_seed),
        alpha_init=0.0,
        enforce_user_served=True,
    )


def build_power_options(args: argparse.Namespace, debug_log_path: Optional[Path]) -> OptimizedPowerOptions:
    return OptimizedPowerOptions(
        qos_target_mode=args.power_qos_target_mode,
        qos_sinr_target=float(args.power_qos_sinr_target),
        qos_kappa=float(args.power_qos_kappa),
        max_dinkelbach_iters=int(args.power_max_dinkelbach_iters),
        max_sca_iters=int(args.power_max_sca_iters),
        dinkelbach_tol=float(args.power_dinkelbach_tol),
        sca_tol=float(args.power_sca_tol),
        qos_tol=float(args.power_qos_tol),
        solver=args.power_solver,
        fallback_solvers=tuple(args.power_fallback_solvers.split(",")) if args.power_fallback_solvers else (),
        solver_verbose=bool(args.power_solver_verbose),
        warm_start=bool(args.power_warm_start),
        debug_enabled=bool(args.power_debug_enabled),
        debug_log_path=(str(debug_log_path) if debug_log_path is not None else None),
        debug_capture_solver_output=bool(args.power_debug_capture_solver_output),
        debug_check_qmat_psd=bool(args.power_debug_check_qmat_psd),
        debug_zero_power_users=bool(args.power_debug_zero_power_users),
        safe_user_power_floor_watt=float(args.power_safe_user_power_floor_watt),
    )


# -----------------------------------------------------------------------------
# Pipeline preparation
# -----------------------------------------------------------------------------


def prepare_upstream_and_pilots(
    cfg: Config,
    *,
    args: argparse.Namespace,
    seed_base: int,
) -> dict[str, Any]:
    seeds = stage_seeds(seed_base)

    geom = generate_geometry_setups(cfg)
    ls = compute_large_scale_setups(cfg=cfg, geom=geom, seed=seeds["large_scale"])
    corr = build_correlation_setups(cfg=cfg, geom=geom, ls=ls, seed=seeds["large_scale"])

    equal_phase_rad = np.deg2rad(float(args.equal_phase_deg))
    phases = build_ris_phase_shifts_setups(
        cfg=cfg,
        mode=args.phase_mode,
        seed=seeds["phase"],
        equal_phase=equal_phase_rad,
    )

    pilot = assign_pilots_setups(
        cfg=cfg,
        geom=geom,
        ls=ls,
        corr=corr,
        phases=phases,
        scheme=args.pilot_scheme,
        Q=int(args.q_neighbors),
        protect_masters=bool(args.protect_masters),
        seed=seeds["pilot"],
    )

    return {
        "seeds": seeds,
        "geom": geom,
        "large_scale": ls,
        "corr": corr,
        "phases": phases,
        "pilot": pilot,
    }


# -----------------------------------------------------------------------------
# Association extraction
# -----------------------------------------------------------------------------


def extract_association_npz(prefix: str, assoc: Any, runtime_seconds: float) -> dict[str, np.ndarray]:
    final_mask = np.asarray(assoc.final_serve_mask, dtype=bool)
    final_avg_serving_aps_per_setup = np.mean(np.sum(final_mask, axis=1), axis=1)  # (S,)

    out: dict[str, np.ndarray] = {
        f"{prefix}_assoc_final_serve_mask": np.asarray(assoc.final_serve_mask, dtype=np.uint8),
        f"{prefix}_assoc_alpha_history": np.asarray(assoc.alpha_history, dtype=float),
        f"{prefix}_assoc_residual_history": np.asarray(assoc.residual_history, dtype=float),
        f"{prefix}_assoc_outer_mean_ee_history": np.asarray(assoc.outer_mean_ee_history, dtype=float),
        f"{prefix}_assoc_outer_mean_se_history": np.asarray(assoc.outer_mean_se_history, dtype=float),
        f"{prefix}_assoc_outer_avg_serving_aps_per_user_history": np.asarray(
            assoc.outer_avg_serving_aps_per_user_history, dtype=float
        ),
        f"{prefix}_assoc_outer_num_users_with_mask_change_history": np.asarray(
            assoc.outer_num_users_with_mask_change_history, dtype=float
        ),
        f"{prefix}_assoc_converged": np.asarray(int(bool(assoc.converged)), dtype=np.int64),
        f"{prefix}_assoc_num_outer_iterations": np.asarray(int(assoc.num_outer_iterations), dtype=np.int64),
        f"{prefix}_assoc_runtime_seconds": np.asarray(float(runtime_seconds), dtype=float),
        f"{prefix}_assoc_runtime_per_setup_seconds": np.asarray(
            float(runtime_seconds) / max(final_mask.shape[0], 1), dtype=float
        ),
        f"{prefix}_assoc_final_ee_bit_per_joule_per_setup": np.asarray(
            assoc.final_ee.ee_bit_per_joule, dtype=float
        ),
        f"{prefix}_assoc_final_ee_mbit_per_joule_per_setup": np.asarray(
            assoc.final_ee.ee_mbit_per_joule, dtype=float
        ),
        f"{prefix}_assoc_final_sum_rate_bps_per_setup": np.asarray(
            assoc.final_ee.sum_rate_bps, dtype=float
        ),
        f"{prefix}_assoc_final_total_power_watt_per_setup": np.asarray(
            assoc.final_ee.total_power_watt, dtype=float
        ),
        f"{prefix}_assoc_final_sum_se_per_setup": np.asarray(
            assoc.final_se.sum_se, dtype=float
        ),
        f"{prefix}_assoc_final_avg_serving_aps_per_user_per_setup": np.asarray(
            final_avg_serving_aps_per_setup, dtype=float
        ),
    }

    # Inner fixed-alpha histories, useful for supplementary diagnostics.
    outer_results = list(assoc.fixed_alpha_results_by_outer_iter)
    outer_count = len(outer_results)
    setup_count = final_mask.shape[0]
    max_inner_len = 0
    for outer in outer_results:
        for r in outer:
            max_inner_len = max(max_inner_len, len(r.transformed_objective_history))
            max_inner_len = max(max_inner_len, len(r.utility_proxy_history))
            max_inner_len = max(max_inner_len, len(r.ee_bit_per_joule_history))
            max_inner_len = max(max_inner_len, len(r.sum_rate_bps_history))
            max_inner_len = max(max_inner_len, len(r.total_power_watt_history))
            max_inner_len = max(max_inner_len, len(r.num_users_changed_history))

    inner_obj = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_utility = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_ee = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_sum_rate = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_total_power = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_changed = np.full((outer_count, setup_count, max_inner_len), np.nan, dtype=float)
    inner_iters = np.full((outer_count, setup_count), np.nan, dtype=float)
    inner_converged = np.full((outer_count, setup_count), np.nan, dtype=float)
    inner_cycle = np.full((outer_count, setup_count), np.nan, dtype=float)

    for o, outer in enumerate(outer_results):
        for s, r in enumerate(outer):
            obj = np.asarray(r.transformed_objective_history, dtype=float)
            util = np.asarray(r.utility_proxy_history, dtype=float)
            ee_hist = np.asarray(r.ee_bit_per_joule_history, dtype=float)
            rate_hist = np.asarray(r.sum_rate_bps_history, dtype=float)
            power_hist = np.asarray(r.total_power_watt_history, dtype=float)
            chg = np.asarray(r.num_users_changed_history, dtype=float)

            inner_obj[o, s, : obj.size] = obj
            inner_utility[o, s, : util.size] = util
            inner_ee[o, s, : ee_hist.size] = ee_hist
            inner_sum_rate[o, s, : rate_hist.size] = rate_hist
            inner_total_power[o, s, : power_hist.size] = power_hist

            # Put changed users from index 1 onward because index 0 is the initial profile.
            if chg.size > 0:
                n_chg = min(chg.size, max_inner_len - 1)
                inner_changed[o, s, 1:n_chg + 1] = chg[:n_chg]
            inner_iters[o, s] = float(r.num_iterations)
            inner_converged[o, s] = float(int(bool(r.converged)))
            inner_cycle[o, s] = float(int(bool(getattr(r, "cycle_detected", False))))

    out[f"{prefix}_assoc_inner_transformed_objective"] = inner_obj
    out[f"{prefix}_assoc_inner_utility_proxy"] = inner_utility
    out[f"{prefix}_assoc_inner_ee_bit_per_joule"] = inner_ee
    out[f"{prefix}_assoc_inner_sum_rate_bps"] = inner_sum_rate
    out[f"{prefix}_assoc_inner_total_power_watt"] = inner_total_power
    out[f"{prefix}_assoc_inner_num_users_changed"] = inner_changed
    out[f"{prefix}_assoc_inner_num_iterations"] = inner_iters
    out[f"{prefix}_assoc_inner_converged"] = inner_converged
    out[f"{prefix}_assoc_inner_cycle_detected"] = inner_cycle
    return out


def summarize_association(prefix: str, npz_entries: dict[str, np.ndarray]) -> dict[str, Any]:
    ee_hist = npz_entries.get(f"{prefix}_assoc_outer_mean_ee_history", np.array([]))
    res_hist = npz_entries.get(f"{prefix}_assoc_residual_history", np.array([]))
    se_hist = npz_entries.get(f"{prefix}_assoc_outer_mean_se_history", np.array([]))
    avg_aps_hist = npz_entries.get(f"{prefix}_assoc_outer_avg_serving_aps_per_user_history", np.array([]))
    changed_hist = npz_entries.get(f"{prefix}_assoc_outer_num_users_with_mask_change_history", np.array([]))

    initial_ee = safe_first(ee_hist)
    final_ee = safe_last(ee_hist)

    return {
        "converged": bool(int(npz_entries[f"{prefix}_assoc_converged"])),
        "num_outer_iterations": int(npz_entries[f"{prefix}_assoc_num_outer_iterations"]),
        "runtime_seconds": float(npz_entries[f"{prefix}_assoc_runtime_seconds"]),
        "runtime_per_setup_seconds": float(npz_entries[f"{prefix}_assoc_runtime_per_setup_seconds"]),
        "initial_mean_ee_bit_per_joule": initial_ee,
        "final_mean_ee_bit_per_joule": final_ee,
        "relative_ee_improvement": rel_improvement(final_ee, initial_ee),
        "final_relative_residual": safe_last(res_hist),
        "final_mean_sum_se": safe_last(se_hist),
        "final_avg_serving_aps_per_user": safe_last(avg_aps_hist),
        "final_num_users_with_mask_change": safe_last(changed_hist),
        "mean_inner_game_iterations": finite_mean(npz_entries.get(f"{prefix}_assoc_inner_num_iterations", np.array([]))),
        "fixed_alpha_game_convergence_fraction": finite_mean(
            npz_entries.get(f"{prefix}_assoc_inner_converged", np.array([]))
        ),
    }


# -----------------------------------------------------------------------------
# Power extraction
# -----------------------------------------------------------------------------


def solver_status_counts(debugs: list[Any]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for d in debugs:
        c.update([str(x) for x in d.solver_status_history])
    return dict(c)


POWER_STOP_REASON_TO_CODE = {
    "unknown": 0,
    "converged": 1,
    "rejected_candidate": 2,
    "hit_max_dinkelbach_iterations": 3,
    "final_qos_infeasible": 4,
    "stopped_without_convergence": 5,
}

POWER_STOP_CODE_TO_REASON = {
    v: k for k, v in POWER_STOP_REASON_TO_CODE.items()
}


def get_power_stop_reason(debug: Any) -> str:
    """
    Return the power optimizer stop reason.

    New runs should have debug.stop_reason from power_allocation.py.
    For older debug objects, infer the most likely reason from existing fields.
    """
    reason = getattr(debug, "stop_reason", None)
    if isinstance(reason, str) and reason:
        return reason

    if bool(getattr(debug, "converged_dinkelbach", False)):
        return "converged"

    if bool(getattr(debug, "stopped_on_rejected_candidate", False)):
        return "rejected_candidate"

    return "stopped_without_convergence"


def power_stop_reason_codes(debugs: list[Any]) -> np.ndarray:
    codes = []
    for d in debugs:
        reason = get_power_stop_reason(d)
        codes.append(POWER_STOP_REASON_TO_CODE.get(reason, 0))
    return np.asarray(codes, dtype=np.int64)


def stop_reason_count_dict_from_codes(codes: np.ndarray) -> dict[str, int]:
    codes = np.asarray(codes, dtype=int).reshape(-1)
    if codes.size == 0:
        return {}
    out: dict[str, int] = {}
    for code in np.unique(codes):
        reason = POWER_STOP_CODE_TO_REASON.get(int(code), "unknown")
        out[reason] = int(np.sum(codes == code))
    return out


def extract_power_npz(
    prefix: str,
    power: Any,
    sinr_cf: Any,
    se_out: Any,
    ee_out: Any,
    runtime_seconds: float,
) -> dict[str, np.ndarray]:
    debugs = list(power.optimized_debug_by_setup or [])
    setup_count = len(debugs)

    alpha_hist = [d.alpha_history for d in debugs]
    residual_hist = [d.dinkelbach_residual_history for d in debugs]
    sca_hist = [d.sca_rho_change_history for d in debugs]
    objective_hist = [d.objective_history for d in debugs]
    qos_margin_hist = [d.min_qos_margin_history for d in debugs]
    min_user_power_hist = [d.min_user_total_power_history for d in debugs]
    below_safe_hist = [d.num_users_below_safe_floor_history for d in debugs]

    final_avg_serving_aps_per_setup = np.mean(np.sum(power.serve_mask_used, axis=1), axis=1)

    status_counts = solver_status_counts(debugs)
    status_keys = np.asarray(list(status_counts.keys()), dtype="U64")
    status_values = np.asarray(list(status_counts.values()), dtype=np.int64)

    stop_reason_code = power_stop_reason_codes(debugs)
    stopped_on_rejected_candidate = np.asarray(
        [int(bool(getattr(d, "stopped_on_rejected_candidate", False))) for d in debugs],
        dtype=np.int64,
    )
    true_hit_max_dinkelbach = np.asarray(
        [int(get_power_stop_reason(d) == "hit_max_dinkelbach_iterations") for d in debugs],
        dtype=np.int64,
    )
    other_nonconverged = np.asarray(
        [
            int(
                (not bool(getattr(d, "converged_dinkelbach", False)))
                and (get_power_stop_reason(d) not in ("rejected_candidate", "hit_max_dinkelbach_iterations"))
            )
            for d in debugs
        ],
        dtype=np.int64,
    )

    out: dict[str, np.ndarray] = {
        f"{prefix}_power_alpha_padded": pad_histories(alpha_hist, mode="last"),
        f"{prefix}_power_alpha_available": pad_histories(alpha_hist, mode="nan"),
        f"{prefix}_power_residual_padded": pad_histories(residual_hist, mode="last"),
        f"{prefix}_power_residual_available": pad_histories(residual_hist, mode="nan"),
        f"{prefix}_power_sca_rho_change_padded": pad_histories(sca_hist, mode="last"),
        f"{prefix}_power_sca_rho_change_available": pad_histories(sca_hist, mode="nan"),
        f"{prefix}_power_objective_padded": pad_histories(objective_hist, mode="last"),
        f"{prefix}_power_objective_available": pad_histories(objective_hist, mode="nan"),
        f"{prefix}_power_min_qos_margin_padded": pad_histories(qos_margin_hist, mode="last"),
        f"{prefix}_power_min_qos_margin_available": pad_histories(qos_margin_hist, mode="nan"),
        f"{prefix}_power_min_user_power_padded": pad_histories(min_user_power_hist, mode="last"),
        f"{prefix}_power_min_user_power_available": pad_histories(min_user_power_hist, mode="nan"),
        f"{prefix}_power_num_users_below_safe_floor_padded": pad_histories(below_safe_hist, mode="last"),
        f"{prefix}_power_num_users_below_safe_floor_available": pad_histories(below_safe_hist, mode="nan"),
        f"{prefix}_power_converged_dinkelbach": np.asarray(
            [int(bool(d.converged_dinkelbach)) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_converged_sca_last_outer": np.asarray(
            [int(bool(d.converged_sca_last_outer)) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_stop_reason_code": stop_reason_code,
        f"{prefix}_power_stopped_on_rejected_candidate": stopped_on_rejected_candidate,
        f"{prefix}_power_true_hit_max_dinkelbach": true_hit_max_dinkelbach,
        f"{prefix}_power_other_nonconverged": other_nonconverged,
        f"{prefix}_power_initial_feasible": np.asarray(
            [int(bool(d.initial_feasible)) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_final_qos_feasible": np.asarray(
            [int(bool(d.final_qos_feasible)) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_safe_point_found": np.asarray(
            [int(bool(d.safe_point_found)) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_num_dinkelbach_iters": np.asarray(
            [int(d.num_dinkelbach_iters) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_num_sca_iters_last_outer": np.asarray(
            [int(d.num_sca_iters_last_outer) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_final_min_qos_margin": np.asarray(
            [float(d.final_min_qos_margin) for d in debugs], dtype=float
        ),
        f"{prefix}_power_final_min_user_total_power": np.asarray(
            [float(d.final_min_user_total_power) for d in debugs], dtype=float
        ),
        f"{prefix}_power_final_num_users_below_safe_floor": np.asarray(
            [int(d.final_num_users_below_safe_floor) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_zero_power_users_before_count": np.asarray(
            [len(d.zero_power_users_before) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_zero_power_users_after_count": np.asarray(
            [len(d.zero_power_users_after) for d in debugs], dtype=np.int64
        ),
        f"{prefix}_power_runtime_seconds": np.asarray(float(runtime_seconds), dtype=float),
        f"{prefix}_power_runtime_per_setup_seconds": np.asarray(
            float(runtime_seconds) / max(setup_count, 1), dtype=float
        ),
        f"{prefix}_power_final_ee_bit_per_joule_per_setup": np.asarray(
            ee_out.ee_bit_per_joule, dtype=float
        ),
        f"{prefix}_power_final_ee_mbit_per_joule_per_setup": np.asarray(
            ee_out.ee_mbit_per_joule, dtype=float
        ),
        f"{prefix}_power_final_sum_rate_bps_per_setup": np.asarray(
            ee_out.sum_rate_bps, dtype=float
        ),
        f"{prefix}_power_final_total_power_watt_per_setup": np.asarray(
            ee_out.total_power_watt, dtype=float
        ),
        f"{prefix}_power_final_sum_se_per_setup": np.asarray(se_out.sum_se, dtype=float),
        f"{prefix}_power_final_avg_serving_aps_per_user_per_setup": np.asarray(
            final_avg_serving_aps_per_setup, dtype=float
        ),
        f"{prefix}_power_solver_status_keys": status_keys,
        f"{prefix}_power_solver_status_counts": status_values,
    }
    return out


def summarize_power(prefix: str, npz_entries: dict[str, np.ndarray]) -> dict[str, Any]:
    alpha = npz_entries.get(f"{prefix}_power_alpha_padded", np.empty((0, 0)))
    residual = npz_entries.get(f"{prefix}_power_residual_padded", np.empty((0, 0)))
    final_ee = npz_entries.get(f"{prefix}_power_final_ee_bit_per_joule_per_setup", np.array([]))
    final_rate = npz_entries.get(f"{prefix}_power_final_sum_rate_bps_per_setup", np.array([]))
    final_power = npz_entries.get(f"{prefix}_power_final_total_power_watt_per_setup", np.array([]))

    if alpha.size > 0:
        initial_alpha = finite_mean(alpha[:, 0])
        final_alpha = finite_mean(alpha[:, -1])
    else:
        initial_alpha = float("nan")
        final_alpha = float("nan")

    if residual.size > 0:
        final_residual = finite_mean(residual[:, -1])
    else:
        final_residual = float("nan")

    converged = npz_entries.get(f"{prefix}_power_converged_dinkelbach", np.array([]))
    qos_ok = npz_entries.get(f"{prefix}_power_final_qos_feasible", np.array([]))
    safe_found = npz_entries.get(f"{prefix}_power_safe_point_found", np.array([]))
    db_iters = npz_entries.get(f"{prefix}_power_num_dinkelbach_iters", np.array([]))
    below_safe = npz_entries.get(f"{prefix}_power_final_num_users_below_safe_floor", np.array([]))
    rejected = npz_entries.get(f"{prefix}_power_stopped_on_rejected_candidate", np.array([]))
    true_hit_max = npz_entries.get(f"{prefix}_power_true_hit_max_dinkelbach", np.array([]))
    other_nonconverged = npz_entries.get(f"{prefix}_power_other_nonconverged", np.array([]))
    stop_reason_code = npz_entries.get(f"{prefix}_power_stop_reason_code", np.array([]))

    status_keys = npz_entries.get(f"{prefix}_power_solver_status_keys", np.array([], dtype="U64"))
    status_counts_arr = npz_entries.get(f"{prefix}_power_solver_status_counts", np.array([], dtype=np.int64))
    status_counts = {str(k): int(v) for k, v in zip(status_keys, status_counts_arr)}

    return {
        "runtime_seconds": float(npz_entries.get(f"{prefix}_power_runtime_seconds", np.asarray(np.nan))),
        "runtime_per_setup_seconds": float(
            npz_entries.get(f"{prefix}_power_runtime_per_setup_seconds", np.asarray(np.nan))
        ),
        "num_setups": int(converged.size),
        "dinkelbach_converged_count": int(np.sum(converged)) if converged.size else 0,
        "dinkelbach_nonconverged_count": int(converged.size - np.sum(converged)) if converged.size else 0,
        "dinkelbach_rejected_candidate_count": int(np.sum(rejected)) if rejected.size else 0,
        "dinkelbach_true_hit_max_count": int(np.sum(true_hit_max)) if true_hit_max.size else 0,
        "dinkelbach_other_nonconverged_count": int(np.sum(other_nonconverged)) if other_nonconverged.size else 0,
        # Backward-compatible name. This now means true max-iteration hit, not every non-converged setup.
        "dinkelbach_hit_cap_count": int(np.sum(true_hit_max)) if true_hit_max.size else 0,
        "stop_reason_counts": stop_reason_count_dict_from_codes(stop_reason_code),
        "mean_num_dinkelbach_iters": finite_mean(db_iters),
        "median_num_dinkelbach_iters": finite_median(db_iters),
        "initial_mean_alpha_equal_power_ee": initial_alpha,
        "final_mean_alpha_ee": final_alpha,
        "relative_alpha_improvement": rel_improvement(final_alpha, initial_alpha),
        "mean_final_residual": final_residual,
        "mean_final_ee_bit_per_joule": finite_mean(final_ee),
        "mean_final_sum_rate_bps": finite_mean(final_rate),
        "mean_final_total_power_watt": finite_mean(final_power),
        "final_qos_feasible_count": int(np.sum(qos_ok)) if qos_ok.size else 0,
        "safe_point_found_count": int(np.sum(safe_found)) if safe_found.size else 0,
        "mean_final_num_users_below_safe_floor": finite_mean(below_safe),
        "solver_status_counts": status_counts,
    }


# -----------------------------------------------------------------------------
# Scenario execution
# -----------------------------------------------------------------------------


def run_scenario_chunk(
    *,
    args: argparse.Namespace,
    scenario: str,
    run_dir: Path,
    chunk_id: int,
    chunk_start: int,
    chunk_num_setups: int,
    chunk_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    prefix = scenario
    lines: list[str] = []
    npz_entries: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "scenario": scenario,
        "status": "started",
        "chunk_seed": int(chunk_seed),
        "chunk_start": int(chunk_start),
        "chunk_num_setups": int(chunk_num_setups),
        "errors": {},
    }

    cfg = build_config_for_scenario(
        args=args,
        scenario=scenario,
        chunk_seed=chunk_seed,
        chunk_num_setups=chunk_num_setups,
    )
    metadata["config"] = asdict(cfg)
    metadata["dimensions"] = {
        "num_setups": int(cfg.sim.num_setups),
        "num_users": int(cfg.dims.num_users),
        "num_aps": int(cfg.dims.num_aps),
        "num_ap_antennas": int(cfg.dims.num_ap_antennas),
        "num_ris": int(cfg.dims.num_ris),
        "num_ris_elements": int(cfg.dims.num_ris_elements) if cfg.dims.num_ris > 0 else 0,
        "tau_c": float(cfg.pilots.coherence_block_length),
        "tau_p": int(cfg.tau_p()),
        "bandwidth_hz": float(cfg.noise.bandwidth_hz),
    }

    lines.append(f"Scenario: {scenario}")
    lines.append("-" * 80)
    lines.append(f"setups in chunk: {chunk_num_setups}")
    lines.append(f"chunk seed: {chunk_seed}")
    lines.append(
        "dims: "
        f"L={cfg.dims.num_aps}, K={cfg.dims.num_users}, M={cfg.dims.num_ap_antennas}, "
        f"T={cfg.dims.num_ris if cfg.ris.enable_ris else 0}, "
        f"N={cfg.dims.num_ris_elements if cfg.ris.enable_ris else 0}"
    )

    prep_t0 = time.perf_counter()
    prepared = prepare_upstream_and_pilots(cfg, args=args, seed_base=chunk_seed)
    prep_seconds = time.perf_counter() - prep_t0
    metadata["upstream_preparation_seconds"] = float(prep_seconds)
    npz_entries[f"{prefix}_upstream_preparation_seconds"] = np.asarray(prep_seconds, dtype=float)
    lines.append(f"upstream preparation seconds: {prep_seconds:.3f}")

    geom = prepared["geom"]
    ls = prepared["large_scale"]
    corr = prepared["corr"]
    phases = prepared["phases"]
    pilot = prepared["pilot"]

    # Association game convergence experiment.
    try:
        game_options = build_game_options(args, random_seed=chunk_seed + 7001)
        assoc_t0 = time.perf_counter()
        assoc = optimize_association_game_ee(
            cfg=cfg,
            geom=geom,
            ls=ls,
            corr=corr,
            phases=phases,
            pilot=pilot,
            game_options=game_options,
        )
        assoc_seconds = time.perf_counter() - assoc_t0
        assoc_entries = extract_association_npz(prefix, assoc, assoc_seconds)
        npz_entries.update(assoc_entries)

        # Write detailed per-chunk association debug files.
        write_association_debug_files(
            run_dir=run_dir,
            chunk_id=chunk_id,
            scenario=scenario,
            assoc=assoc,
        )

        assoc_summary = summarize_association(prefix, npz_entries)
        metadata["association"] = assoc_summary
        lines.append("")
        lines.append("Association game optimizer")
        lines.append(f"  converged: {assoc_summary['converged']}")
        lines.append(f"  outer iterations: {assoc_summary['num_outer_iterations']}")
        lines.append(f"  runtime seconds: {assoc_summary['runtime_seconds']:.3f}")
        lines.append(f"  runtime per setup seconds: {assoc_summary['runtime_per_setup_seconds']:.3f}")
        lines.append(f"  initial mean EE bit/J: {assoc_summary['initial_mean_ee_bit_per_joule']:.6e}")
        lines.append(f"  final mean EE bit/J: {assoc_summary['final_mean_ee_bit_per_joule']:.6e}")
        lines.append(f"  relative EE improvement: {assoc_summary['relative_ee_improvement']:.6e}")
        lines.append(f"  final relative residual: {assoc_summary['final_relative_residual']:.6e}")
        lines.append(f"  final avg serving APs/user: {assoc_summary['final_avg_serving_aps_per_user']:.3f}")
        lines.append(f"  final users with changed masks: {assoc_summary['final_num_users_with_mask_change']:.0f}")
    except Exception as exc:  # keep chunk output even if association fails
        metadata["errors"]["association"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        lines.append("")
        lines.append("Association game optimizer FAILED")
        lines.append(traceback.format_exc())

    # Optimized power allocation convergence experiment on association-optimized sets.
    try:
        est_stats = build_channel_estimation_statistics(
            cfg=cfg,
            corr=corr,
            ls=ls,
            pilot_of_user=pilot.pilot_of_user,
            pilot_groups=pilot.pilot_groups,
            phases=phases,
        )

        debug_log_path = None
        if args.power_debug_enabled:
            debug_log_path = run_dir / "chunks" / "logs" / f"power_debug_{scenario}_chunk_{chunk_id:03d}.txt"
            ensure_dir(debug_log_path.parent)

        power_options = build_power_options(args, debug_log_path=debug_log_path)

        optimized_mask = np.asarray(assoc.final_serve_mask, dtype=bool)
        power_t0 = time.perf_counter()
        power_out = allocate_downlink_power(
            cfg,
            scheme="optimized",
            serve_mask=optimized_mask,
            est=est_stats,
            corr=corr,
            ls=ls,
            pilot_of_user=pilot.pilot_of_user,
            phases=phases,
            optimized_options=power_options,
        )
        power_seconds = time.perf_counter() - power_t0

        sinr_cf = compute_dl_sinr_closed_form(
            cfg=cfg,
            est=est_stats,
            corr=corr,
            ls=ls,
            power=power_out,
            pilot_of_user=pilot.pilot_of_user,
            phases=phases,
            serve_mask=optimized_mask,
        )
        se_out = compute_spectral_efficiency_setups(cfg, sinr_cf.sinr_dl)
        ee_out = compute_energy_efficiency_setups(cfg, se_out, power_out, serve_mask=optimized_mask)

        power_entries = extract_power_npz(prefix, power_out, sinr_cf, se_out, ee_out, power_seconds)
        npz_entries.update(power_entries)
        power_summary = summarize_power(prefix, npz_entries)
        metadata["power"] = power_summary
        lines.append("")
        lines.append("Optimized power allocation on association-optimized sets")
        lines.append(f"  setups: {power_summary['num_setups']}")
        lines.append(f"  Dinkelbach converged count: {power_summary['dinkelbach_converged_count']}")
        lines.append(f"  Dinkelbach nonconverged count: {power_summary['dinkelbach_nonconverged_count']}")
        lines.append(f"  Dinkelbach rejected-candidate stops: {power_summary['dinkelbach_rejected_candidate_count']}")
        lines.append(f"  Dinkelbach true hit-max count: {power_summary['dinkelbach_true_hit_max_count']}")
        lines.append(f"  Dinkelbach other nonconverged count: {power_summary['dinkelbach_other_nonconverged_count']}")
        lines.append(f"  stop reason counts: {power_summary['stop_reason_counts']}")
        lines.append(f"  runtime seconds: {power_summary['runtime_seconds']:.3f}")
        lines.append(f"  runtime per setup seconds: {power_summary['runtime_per_setup_seconds']:.3f}")
        lines.append(f"  initial mean alpha/EE bit/J: {power_summary['initial_mean_alpha_equal_power_ee']:.6e}")
        lines.append(f"  final mean alpha/EE bit/J: {power_summary['final_mean_alpha_ee']:.6e}")
        lines.append(f"  relative alpha improvement: {power_summary['relative_alpha_improvement']:.6e}")
        lines.append(f"  mean final residual: {power_summary['mean_final_residual']:.6e}")
        lines.append(f"  final QoS feasible count: {power_summary['final_qos_feasible_count']}")
        lines.append(f"  safe point found count: {power_summary['safe_point_found_count']}")
        lines.append(f"  mean users below safe floor: {power_summary['mean_final_num_users_below_safe_floor']:.3f}")
        lines.append(f"  solver status counts: {power_summary['solver_status_counts']}")
    except Exception as exc:  # keep chunk output even if power optimization fails
        metadata["errors"]["power"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        lines.append("")
        lines.append("Optimized power allocation FAILED")
        lines.append(traceback.format_exc())

    metadata["status"] = "ok" if not metadata["errors"] else "partial"
    lines.append("")
    return npz_entries, metadata, lines


# -----------------------------------------------------------------------------
# Chunk mode
# -----------------------------------------------------------------------------


def run_chunk(args: argparse.Namespace) -> None:
    run_dir = build_run_dir(args)
    chunk_dir = run_dir / "chunks"
    ensure_dir(chunk_dir)

    if args.chunk_id is None:
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        else:
            chunk_id = 0
    else:
        chunk_id = int(args.chunk_id)

    chunk_start, chunk_end, chunk_num_setups = compute_chunk_range(
        int(args.total_setups), int(args.total_chunks), chunk_id
    )
    chunk_seed = int(args.base_seed) + 10000 * int(chunk_id)

    npz_entries: dict[str, np.ndarray] = {
        "chunk_id": np.asarray(chunk_id, dtype=np.int64),
        "chunk_start": np.asarray(chunk_start, dtype=np.int64),
        "chunk_end": np.asarray(chunk_end, dtype=np.int64),
        "chunk_num_setups": np.asarray(chunk_num_setups, dtype=np.int64),
        "chunk_seed": np.asarray(chunk_seed, dtype=np.int64),
    }

    metadata: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "run_chunk",
        "tag": args.tag,
        "run_dir": str(run_dir),
        "chunk_dir": str(chunk_dir),
        "chunk_id": int(chunk_id),
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_end),
        "chunk_num_setups": int(chunk_num_setups),
        "total_setups": int(args.total_setups),
        "total_chunks": int(args.total_chunks),
        "base_seed": int(args.base_seed),
        "chunk_seed": int(chunk_seed),
        "is_slurm": is_slurm_environment(),
        "slurm": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith("SLURM_")},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "argv": sys.argv,
        "args": vars(args),
        "scenarios": {},
    }

    summary_lines: list[str] = []
    summary_lines.append("Convergence behavior chunk summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"run_dir: {run_dir}")
    summary_lines.append(f"chunk_id: {chunk_id}")
    summary_lines.append(f"setups: [{chunk_start}, {chunk_end}) count={chunk_num_setups}")
    summary_lines.append(f"chunk_seed: {chunk_seed}")
    summary_lines.append(f"hostname: {socket.gethostname()}")
    summary_lines.append("")

    chunk_t0 = time.perf_counter()
    for scenario in SCENARIOS:
        # Use the same seed for no-RIS and single-RIS so AP/user geometry and
        # large-scale randomness are matched as much as possible. The scenario
        # override still controls whether RIS is enabled.
        scenario_seed = chunk_seed
        scenario_entries, scenario_meta, scenario_lines = run_scenario_chunk(
            args=args,
            scenario=scenario,
            run_dir=run_dir,
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_num_setups=chunk_num_setups,
            chunk_seed=scenario_seed,
        )
        npz_entries.update(scenario_entries)
        metadata["scenarios"][scenario] = scenario_meta
        summary_lines.extend(scenario_lines)
        summary_lines.append("")

    total_seconds = time.perf_counter() - chunk_t0
    metadata["chunk_total_runtime_seconds"] = float(total_seconds)
    npz_entries["chunk_total_runtime_seconds"] = np.asarray(total_seconds, dtype=float)
    summary_lines.append(f"Total chunk runtime seconds: {total_seconds:.3f}")

    npz_path = chunk_dir / f"data_chunk_{chunk_id:03d}.npz"
    json_path = chunk_dir / f"metadata_chunk_{chunk_id:03d}.json"
    txt_path = chunk_dir / f"summary_chunk_{chunk_id:03d}.txt"

    np.savez_compressed(npz_path, **npz_entries)
    write_json(json_path, metadata)
    write_text(txt_path, "\n".join(summary_lines) + "\n")

    print(f"[OK] wrote chunk NPZ: {npz_path}")
    print(f"[OK] wrote chunk JSON: {json_path}")
    print(f"[OK] wrote chunk TXT: {txt_path}")


# -----------------------------------------------------------------------------
# Merge mode
# -----------------------------------------------------------------------------


def collect_chunk_files(run_dir: Path, expected_chunks: Optional[int]) -> list[Path]:
    chunk_dir = run_dir / "chunks"
    files = sorted(chunk_dir.glob("data_chunk_*.npz"))
    if expected_chunks is not None and len(files) != int(expected_chunks):
        print(
            f"[WARNING] expected {expected_chunks} chunk files, found {len(files)} in {chunk_dir}",
            file=sys.stderr,
        )
    if not files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")
    return files


def load_chunk_metadata(npz_path: Path) -> dict[str, Any]:
    json_path = npz_path.with_name(npz_path.name.replace("data_", "metadata_").replace(".npz", ".json"))
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {"missing_metadata_for": str(npz_path)}


def merge_assoc_histories(
    chunks: list[np.lib.npyio.NpzFile],
    weights: np.ndarray,
    scenario: str,
    hist_name: str,
    *,
    pad_mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    used_weights: list[float] = []
    key = f"{scenario}_{hist_name}"
    for data, w in zip(chunks, weights):
        arr = npz_get(data, key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        arrays.append(arr)
        used_weights.append(float(w))
    if not arrays:
        return np.array([]), np.empty((0, 0)), np.array([], dtype=int)
    width = max(a.size for a in arrays)
    padded = np.vstack([pad_1d(a, width, mode=pad_mode) for a in arrays])
    mean, count = weighted_nanmean_rows(padded, np.asarray(used_weights, dtype=float))
    return mean, padded, count


def merge_power_matrices(
    chunks: list[np.lib.npyio.NpzFile],
    scenario: str,
    matrix_name: str,
    *,
    pad_mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats: list[np.ndarray] = []
    key = f"{scenario}_{matrix_name}"
    for data in chunks:
        mat = npz_get(data, key)
        if mat is None:
            continue
        mat = np.asarray(mat, dtype=float)
        if mat.size == 0:
            continue
        if mat.ndim == 1:
            mat = mat[None, :]
        mats.append(mat)
    if not mats:
        return np.array([]), np.empty((0, 0)), np.array([], dtype=int)
    width = max(m.shape[1] for m in mats)
    padded = np.vstack([pad_matrix_columns(m, width, mode=pad_mode) for m in mats])
    mean = np.nanmean(padded, axis=0)
    count = np.sum(np.isfinite(padded), axis=0).astype(int)
    return mean, padded, count


def concat_per_setup(chunks: list[np.lib.npyio.NpzFile], scenario: str, name: str) -> np.ndarray:
    key = f"{scenario}_{name}"
    arrays = []
    for data in chunks:
        arr = npz_get(data, key)
        if arr is not None:
            arrays.append(np.asarray(arr))
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def merge_status_counts(chunks: list[np.lib.npyio.NpzFile], scenario: str) -> dict[str, int]:
    c: Counter[str] = Counter()
    for data in chunks:
        keys = npz_get(data, f"{scenario}_power_solver_status_keys")
        vals = npz_get(data, f"{scenario}_power_solver_status_counts")
        if keys is None or vals is None:
            continue
        for k, v in zip(keys, vals):
            c[str(k)] += int(v)
    return dict(c)


def summarize_merged_scenario(scenario: str, merged: dict[str, np.ndarray], status_counts: dict[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    assoc_ee = merged.get(f"{scenario}_assoc_outer_mean_ee_history_mean_padded", np.array([]))
    assoc_res = merged.get(f"{scenario}_assoc_residual_history_mean_padded", np.array([]))
    assoc_se = merged.get(f"{scenario}_assoc_outer_mean_se_history_mean_padded", np.array([]))
    assoc_aps = merged.get(f"{scenario}_assoc_outer_avg_serving_aps_per_user_history_mean_padded", np.array([]))
    assoc_conv = merged.get(f"{scenario}_assoc_converged_all", np.array([]))
    assoc_iters = merged.get(f"{scenario}_assoc_num_outer_iterations_all", np.array([]))
    assoc_runtime_per_setup = merged.get(f"{scenario}_assoc_runtime_per_setup_seconds_all", np.array([]))

    out["association"] = {
        "chunks": int(assoc_conv.size),
        "converged_chunks": int(np.sum(assoc_conv)) if assoc_conv.size else 0,
        "hit_cap_chunks": int(assoc_conv.size - np.sum(assoc_conv)) if assoc_conv.size else 0,
        "mean_outer_iterations": finite_mean(assoc_iters),
        "median_outer_iterations": finite_median(assoc_iters),
        "mean_runtime_per_setup_seconds": finite_mean(assoc_runtime_per_setup),
        "median_runtime_per_setup_seconds": finite_median(assoc_runtime_per_setup),
        "initial_mean_ee_bit_per_joule": safe_first(assoc_ee),
        "final_mean_ee_bit_per_joule": safe_last(assoc_ee),
        "relative_ee_improvement": rel_improvement(safe_last(assoc_ee), safe_first(assoc_ee)),
        "final_mean_residual": safe_last(assoc_res),
        "final_mean_sum_se": safe_last(assoc_se),
        "final_avg_serving_aps_per_user": safe_last(assoc_aps),
    }

    power_alpha = merged.get(f"{scenario}_power_alpha_padded_mean", np.array([]))
    power_res = merged.get(f"{scenario}_power_residual_padded_mean", np.array([]))
    power_conv = merged.get(f"{scenario}_power_converged_dinkelbach_all", np.array([]))
    power_stop_reason_code = merged.get(f"{scenario}_power_stop_reason_code_all", np.array([]))
    power_rejected = merged.get(f"{scenario}_power_stopped_on_rejected_candidate_all", np.array([]))
    power_true_hit_max = merged.get(f"{scenario}_power_true_hit_max_dinkelbach_all", np.array([]))
    power_other_nonconverged = merged.get(f"{scenario}_power_other_nonconverged_all", np.array([]))
    power_iters = merged.get(f"{scenario}_power_num_dinkelbach_iters_all", np.array([]))
    power_runtime_per_setup = merged.get(f"{scenario}_power_runtime_per_setup_seconds_all", np.array([]))
    power_qos = merged.get(f"{scenario}_power_final_qos_feasible_all", np.array([]))
    power_safe = merged.get(f"{scenario}_power_safe_point_found_all", np.array([]))
    power_below = merged.get(f"{scenario}_power_final_num_users_below_safe_floor_all", np.array([]))
    power_final_rate = merged.get(f"{scenario}_power_final_sum_rate_bps_per_setup_all", np.array([]))
    power_final_total_power = merged.get(f"{scenario}_power_final_total_power_watt_per_setup_all", np.array([]))
    power_final_aps = merged.get(f"{scenario}_power_final_avg_serving_aps_per_user_per_setup_all", np.array([]))

    out["power"] = {
        "setups": int(power_conv.size),
        "dinkelbach_converged_setups": int(np.sum(power_conv)) if power_conv.size else 0,
        "dinkelbach_nonconverged_setups": int(power_conv.size - np.sum(power_conv)) if power_conv.size else 0,
        "dinkelbach_rejected_candidate_setups": int(np.sum(power_rejected)) if power_rejected.size else 0,
        "dinkelbach_true_hit_max_setups": int(np.sum(power_true_hit_max)) if power_true_hit_max.size else 0,
        "dinkelbach_other_nonconverged_setups": int(np.sum(power_other_nonconverged)) if power_other_nonconverged.size else 0,
        # Backward-compatible name. This now means true max-iteration hit, not every non-converged setup.
        "dinkelbach_hit_cap_setups": int(np.sum(power_true_hit_max)) if power_true_hit_max.size else 0,
        "stop_reason_counts": stop_reason_count_dict_from_codes(power_stop_reason_code),
        "mean_num_dinkelbach_iters": finite_mean(power_iters),
        "median_num_dinkelbach_iters": finite_median(power_iters),
        "mean_runtime_per_setup_seconds": finite_mean(power_runtime_per_setup),
        "median_runtime_per_setup_seconds": finite_median(power_runtime_per_setup),
        "initial_mean_alpha_equal_power_ee": safe_first(power_alpha),
        "final_mean_alpha_ee": safe_last(power_alpha),
        "relative_alpha_improvement": rel_improvement(safe_last(power_alpha), safe_first(power_alpha)),
        "final_mean_residual": safe_last(power_res),
        "final_qos_feasible_setups": int(np.sum(power_qos)) if power_qos.size else 0,
        "safe_point_found_setups": int(np.sum(power_safe)) if power_safe.size else 0,
        "mean_final_num_users_below_safe_floor": finite_mean(power_below),
        "mean_final_sum_rate_bps": finite_mean(power_final_rate),
        "mean_final_total_power_watt": finite_mean(power_final_total_power),
        "final_avg_serving_aps_per_user": finite_mean(power_final_aps),
        "solver_status_counts": status_counts,
    }
    return out


def _concat_debug_arrays(debug_npzs: list[np.lib.npyio.NpzFile], scenario: str, name: str) -> np.ndarray:
    key = f"{scenario}_{name}"
    arrays = []
    for d in debug_npzs:
        if key in d.files:
            arrays.append(np.asarray(d[key]))
    if not arrays:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def build_merged_association_debug_report(run_dir: Path) -> None:
    """
    Merge per-chunk association debug files into one readable summary.

    This creates:
        association_debug_summary.txt
        association_debug_summary.json
    """
    debug_dir = run_dir / "chunks" / "association_debug"
    if not debug_dir.exists():
        return

    report_json: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "debug_dir": str(debug_dir),
        "scenarios": {},
    }

    lines: list[str] = []
    lines.append("Association debug summary")
    lines.append("=" * 80)
    lines.append(f"debug_dir: {debug_dir}")
    lines.append("")

    for scenario in SCENARIOS:
        files = sorted(debug_dir.glob(f"{scenario}_association_debug_chunk_*.npz"))
        if not files:
            report_json["scenarios"][scenario] = {"available": False, "files": 0}
            lines.append(f"Scenario: {scenario}")
            lines.append("-" * 80)
            lines.append("No debug files found.")
            lines.append("")
            continue

        debug_npzs = [np.load(p, allow_pickle=False) for p in files]

        candidate_sizes = _concat_debug_arrays(debug_npzs, scenario, "debug_candidate_sizes")
        action_counts = _concat_debug_arrays(debug_npzs, scenario, "debug_action_counts")
        initial_counts = _concat_debug_arrays(debug_npzs, scenario, "debug_initial_serving_counts")
        final_counts = _concat_debug_arrays(debug_npzs, scenario, "debug_final_serving_counts")
        frac_changed = _concat_debug_arrays(debug_npzs, scenario, "debug_fraction_users_changed_per_setup")
        hamming = _concat_debug_arrays(debug_npzs, scenario, "debug_hamming_initial_final_per_setup")
        terminal_differs = _concat_debug_arrays(debug_npzs, scenario, "debug_terminal_differs_from_returned")
        best_ee_differs = _concat_debug_arrays(debug_npzs, scenario, "debug_best_global_ee_differs_from_returned")
        returned_ee = _concat_debug_arrays(debug_npzs, scenario, "debug_returned_ee_bit_per_joule")
        best_ee = _concat_debug_arrays(debug_npzs, scenario, "debug_best_global_ee_bit_per_joule")
        best_ee_gain = _concat_debug_arrays(debug_npzs, scenario, "debug_best_global_ee_gain_over_returned")
        cycle = _concat_debug_arrays(debug_npzs, scenario, "debug_inner_cycle_detected")
        inner_iters = _concat_debug_arrays(debug_npzs, scenario, "debug_inner_iterations")

        summary = {
            "available": True,
            "files": len(files),
            "setups": int(candidate_sizes.shape[0]) if candidate_sizes.size else 0,
            "candidate_aps_per_user_mean": finite_mean(candidate_sizes),
            "candidate_aps_per_user_min": int(np.min(candidate_sizes)) if candidate_sizes.size else None,
            "candidate_aps_per_user_max": int(np.max(candidate_sizes)) if candidate_sizes.size else None,
            "actions_per_user_mean": finite_mean(action_counts),
            "actions_per_user_min": int(np.min(action_counts)) if action_counts.size else None,
            "actions_per_user_max": int(np.max(action_counts)) if action_counts.size else None,
            "users_with_one_action_only": int(np.sum(action_counts <= 1)) if action_counts.size else 0,
            "initial_serving_aps_per_user_mean": finite_mean(initial_counts),
            "final_serving_aps_per_user_mean": finite_mean(final_counts),
            "fraction_users_changed_mean": finite_mean(frac_changed),
            "hamming_initial_final_mean": finite_mean(hamming),
            "terminal_differs_from_returned_setups": int(np.sum(terminal_differs)) if terminal_differs.size else 0,
            "best_global_ee_differs_from_returned_setups": int(np.sum(best_ee_differs)) if best_ee_differs.size else 0,
            "returned_ee_mean_bit_per_joule": finite_mean(returned_ee),
            "best_global_ee_seen_mean_bit_per_joule": finite_mean(best_ee),
            "best_global_ee_gain_over_returned_mean": finite_mean(best_ee_gain),
            "cycle_detected_count": int(np.nansum(cycle)) if cycle.size else 0,
            "mean_inner_iterations": finite_mean(inner_iters),
        }

        report_json["scenarios"][scenario] = summary

        lines.append(f"Scenario: {scenario}")
        lines.append("-" * 80)
        lines.append(f"debug files: {summary['files']}")
        lines.append(f"setups: {summary['setups']}")
        lines.append(f"candidate APs/user mean: {summary['candidate_aps_per_user_mean']}")
        lines.append(f"candidate APs/user min/max: {summary['candidate_aps_per_user_min']} / {summary['candidate_aps_per_user_max']}")
        lines.append(f"actions/user mean: {summary['actions_per_user_mean']}")
        lines.append(f"actions/user min/max: {summary['actions_per_user_min']} / {summary['actions_per_user_max']}")
        lines.append(f"users with one action only: {summary['users_with_one_action_only']}")
        lines.append(f"initial serving APs/user mean: {summary['initial_serving_aps_per_user_mean']}")
        lines.append(f"final serving APs/user mean: {summary['final_serving_aps_per_user_mean']}")
        lines.append(f"fraction users changed initial-to-final mean: {summary['fraction_users_changed_mean']}")
        lines.append(f"hamming initial-to-final mean: {summary['hamming_initial_final_mean']}")
        lines.append(f"terminal differs from returned setups: {summary['terminal_differs_from_returned_setups']}")
        lines.append(f"best-global-EE differs from returned setups: {summary['best_global_ee_differs_from_returned_setups']}")
        lines.append(f"returned EE mean bit/J: {summary['returned_ee_mean_bit_per_joule']}")
        lines.append(f"best global EE seen mean bit/J: {summary['best_global_ee_seen_mean_bit_per_joule']}")
        lines.append(f"best global EE gain over returned mean: {summary['best_global_ee_gain_over_returned_mean']}")
        lines.append(f"cycle detected count: {summary['cycle_detected_count']}")
        lines.append(f"mean inner iterations: {summary['mean_inner_iterations']}")
        lines.append("")

        for d in debug_npzs:
            d.close()

    write_json(run_dir / "association_debug_summary.json", report_json)
    write_text(run_dir / "association_debug_summary.txt", "\n".join(lines) + "\n")


def merge_chunks(args: argparse.Namespace) -> None:
    run_dir = build_run_dir(args)
    ensure_dir(run_dir)
    files = collect_chunk_files(run_dir, args.expected_chunks)

    chunk_npzs = [np.load(p, allow_pickle=False) for p in files]
    chunk_metas = [load_chunk_metadata(p) for p in files]
    weights = np.asarray([float(npz_get(d, "chunk_num_setups", np.asarray(1))) for d in chunk_npzs], dtype=float)

    merged: dict[str, np.ndarray] = {}
    merged_meta: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "merge",
        "tag": args.tag,
        "run_dir": str(run_dir),
        "num_chunk_files_found": len(files),
        "expected_chunks": args.expected_chunks,
        "chunk_files": [str(p) for p in files],
        "chunk_metadata": chunk_metas,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "argv": sys.argv,
        "args": vars(args),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        # Association histories, one trajectory per chunk.
        assoc_histories = [
            "assoc_alpha_history",
            "assoc_residual_history",
            "assoc_outer_mean_ee_history",
            "assoc_outer_mean_se_history",
            "assoc_outer_avg_serving_aps_per_user_history",
            "assoc_outer_num_users_with_mask_change_history",
        ]
        for hist in assoc_histories:
            mean, matrix, count = merge_assoc_histories(
                chunk_npzs, weights, scenario, hist, pad_mode="last"
            )
            merged[f"{scenario}_{hist}_mean_padded"] = mean
            merged[f"{scenario}_{hist}_by_chunk_padded"] = matrix
            merged[f"{scenario}_{hist}_available_count"] = count

            mean_avail, matrix_avail, count_avail = merge_assoc_histories(
                chunk_npzs, weights, scenario, hist, pad_mode="nan"
            )
            merged[f"{scenario}_{hist}_mean_available"] = mean_avail
            merged[f"{scenario}_{hist}_by_chunk_available"] = matrix_avail
            merged[f"{scenario}_{hist}_available_count_unpadded"] = count_avail

        # Association scalar and per-setup outputs.
        for scalar_name in [
            "assoc_converged",
            "assoc_num_outer_iterations",
            "assoc_runtime_seconds",
            "assoc_runtime_per_setup_seconds",
        ]:
            arrs = []
            key = f"{scenario}_{scalar_name}"
            for data in chunk_npzs:
                val = npz_get(data, key)
                if val is not None:
                    arrs.append(np.asarray(val).reshape(1))
            merged[f"{scenario}_{scalar_name}_all"] = np.concatenate(arrs) if arrs else np.array([])

        for name in [
            "assoc_final_ee_bit_per_joule_per_setup",
            "assoc_final_ee_mbit_per_joule_per_setup",
            "assoc_final_sum_rate_bps_per_setup",
            "assoc_final_total_power_watt_per_setup",
            "assoc_final_sum_se_per_setup",
            "assoc_final_avg_serving_aps_per_user_per_setup",
        ]:
            merged[f"{scenario}_{name}_all"] = concat_per_setup(chunk_npzs, scenario, name)

        # Power matrices, one row per setup.
        power_matrices = [
            "power_alpha_padded",
            "power_alpha_available",
            "power_residual_padded",
            "power_residual_available",
            "power_sca_rho_change_padded",
            "power_sca_rho_change_available",
            "power_objective_padded",
            "power_objective_available",
            "power_min_qos_margin_padded",
            "power_min_qos_margin_available",
            "power_min_user_power_padded",
            "power_min_user_power_available",
            "power_num_users_below_safe_floor_padded",
            "power_num_users_below_safe_floor_available",
        ]
        for name in power_matrices:
            pad_mode = "nan" if name.endswith("available") else "last"
            mean, matrix, count = merge_power_matrices(chunk_npzs, scenario, name, pad_mode=pad_mode)
            merged[f"{scenario}_{name}_mean"] = mean
            merged[f"{scenario}_{name}_matrix"] = matrix
            merged[f"{scenario}_{name}_count"] = count

        # Power per-setup scalar arrays.
        for name in [
            "power_converged_dinkelbach",
            "power_converged_sca_last_outer",
            "power_stop_reason_code",
            "power_stopped_on_rejected_candidate",
            "power_true_hit_max_dinkelbach",
            "power_other_nonconverged",
            "power_initial_feasible",
            "power_final_qos_feasible",
            "power_safe_point_found",
            "power_num_dinkelbach_iters",
            "power_num_sca_iters_last_outer",
            "power_final_min_qos_margin",
            "power_final_min_user_total_power",
            "power_final_num_users_below_safe_floor",
            "power_zero_power_users_before_count",
            "power_zero_power_users_after_count",
            "power_final_ee_bit_per_joule_per_setup",
            "power_final_ee_mbit_per_joule_per_setup",
            "power_final_sum_rate_bps_per_setup",
            "power_final_total_power_watt_per_setup",
            "power_final_sum_se_per_setup",
            "power_final_avg_serving_aps_per_user_per_setup",
        ]:
            merged[f"{scenario}_{name}_all"] = concat_per_setup(chunk_npzs, scenario, name)

        # Runtime values are one scalar per chunk.
        for scalar_name in ["power_runtime_seconds", "power_runtime_per_setup_seconds"]:
            arrs = []
            key = f"{scenario}_{scalar_name}"
            for data in chunk_npzs:
                val = npz_get(data, key)
                if val is not None:
                    arrs.append(np.asarray(val).reshape(1))
            merged[f"{scenario}_{scalar_name}_all"] = np.concatenate(arrs) if arrs else np.array([])

        status_counts = merge_status_counts(chunk_npzs, scenario)
        merged[f"{scenario}_power_solver_status_keys_merged"] = np.asarray(list(status_counts.keys()), dtype="U64")
        merged[f"{scenario}_power_solver_status_counts_merged"] = np.asarray(list(status_counts.values()), dtype=np.int64)

        merged_meta["scenarios"][scenario] = summarize_merged_scenario(scenario, merged, status_counts)

    # Add plotting guidance metadata.
    merged_meta["recommended_main_plot_keys"] = {
        "association_figure": {
            "x": "outer Dinkelbach iteration index",
            "subplot_a": [
                "no_ris_assoc_outer_mean_ee_history_mean_padded",
                "single_ris_assoc_outer_mean_ee_history_mean_padded",
            ],
            "subplot_b_log_y": [
                "no_ris_assoc_residual_history_mean_padded",
                "single_ris_assoc_residual_history_mean_padded",
            ],
        },
        "power_figure": {
            "x": "outer Dinkelbach iteration index",
            "subplot_a": [
                "no_ris_power_alpha_padded_mean",
                "single_ris_power_alpha_padded_mean",
            ],
            "subplot_b_log_y": [
                "no_ris_power_residual_padded_mean",
                "single_ris_power_residual_padded_mean",
            ],
        },
        "supplementary": [
            "*_power_sca_rho_change_available_mean",
            "*_power_min_qos_margin_available_mean",
            "*_power_min_user_power_available_mean",
            "*_assoc_outer_num_users_with_mask_change_history_mean_available",
        ],
    }

    merged_npz_path = run_dir / "merged_data.npz"
    merged_json_path = run_dir / "merged_metadata.json"
    merged_txt_path = run_dir / "merged_summary.txt"

    np.savez_compressed(merged_npz_path, **merged)
    write_json(merged_json_path, merged_meta)
    write_text(merged_txt_path, build_merged_summary_text(run_dir, files, merged_meta))

    # Merge association debug files into a readable report.
    build_merged_association_debug_report(run_dir)

    for d in chunk_npzs:
        d.close()

    print(f"[OK] wrote merged NPZ: {merged_npz_path}")
    print(f"[OK] wrote merged JSON: {merged_json_path}")
    print(f"[OK] wrote merged TXT: {merged_txt_path}")


def build_merged_summary_text(run_dir: Path, files: list[Path], meta: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Merged convergence behavior summary")
    lines.append("=" * 80)
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"chunk files found: {len(files)}")
    lines.append(f"created_at: {meta.get('created_at')}")
    lines.append("")

    for scenario in SCENARIOS:
        s_meta = meta.get("scenarios", {}).get(scenario, {})
        lines.append(f"Scenario: {scenario}")
        lines.append("-" * 80)

        assoc = s_meta.get("association", {})
        lines.append("Association game optimizer")
        lines.append(f"  chunks: {assoc.get('chunks')}")
        lines.append(f"  converged chunks: {assoc.get('converged_chunks')}")
        lines.append(f"  hit-cap chunks: {assoc.get('hit_cap_chunks')}")
        lines.append(f"  mean outer iterations: {assoc.get('mean_outer_iterations')}")
        lines.append(f"  median outer iterations: {assoc.get('median_outer_iterations')}")
        lines.append(f"  mean runtime/setup seconds: {assoc.get('mean_runtime_per_setup_seconds')}")
        lines.append(f"  median runtime/setup seconds: {assoc.get('median_runtime_per_setup_seconds')}")
        lines.append(f"  initial mean EE bit/J: {assoc.get('initial_mean_ee_bit_per_joule')}")
        lines.append(f"  final mean EE bit/J: {assoc.get('final_mean_ee_bit_per_joule')}")
        lines.append(f"  relative EE improvement: {assoc.get('relative_ee_improvement')}")
        lines.append(f"  final mean residual: {assoc.get('final_mean_residual')}")
        lines.append(f"  final mean sum SE: {assoc.get('final_mean_sum_se')}")
        lines.append(f"  final avg serving APs/user: {assoc.get('final_avg_serving_aps_per_user')}")
        lines.append("")

        power = s_meta.get("power", {})
        lines.append("Optimized power allocation on association-optimized sets")
        lines.append(f"  setups: {power.get('setups')}")
        lines.append(f"  Dinkelbach converged setups: {power.get('dinkelbach_converged_setups')}")
        lines.append(f"  Dinkelbach nonconverged setups: {power.get('dinkelbach_nonconverged_setups')}")
        lines.append(f"  Dinkelbach rejected-candidate stops: {power.get('dinkelbach_rejected_candidate_setups')}")
        lines.append(f"  Dinkelbach true hit-max setups: {power.get('dinkelbach_true_hit_max_setups')}")
        lines.append(f"  Dinkelbach other nonconverged setups: {power.get('dinkelbach_other_nonconverged_setups')}")
        lines.append(f"  stop reason counts: {power.get('stop_reason_counts')}")
        lines.append(f"  mean Dinkelbach iterations: {power.get('mean_num_dinkelbach_iters')}")
        lines.append(f"  median Dinkelbach iterations: {power.get('median_num_dinkelbach_iters')}")
        lines.append(f"  mean runtime/setup seconds: {power.get('mean_runtime_per_setup_seconds')}")
        lines.append(f"  median runtime/setup seconds: {power.get('median_runtime_per_setup_seconds')}")
        lines.append(f"  initial mean alpha/EE bit/J: {power.get('initial_mean_alpha_equal_power_ee')}")
        lines.append(f"  final mean alpha/EE bit/J: {power.get('final_mean_alpha_ee')}")
        lines.append(f"  relative alpha improvement: {power.get('relative_alpha_improvement')}")
        lines.append(f"  final mean residual: {power.get('final_mean_residual')}")
        lines.append(f"  final QoS feasible setups: {power.get('final_qos_feasible_setups')}")
        lines.append(f"  safe point found setups: {power.get('safe_point_found_setups')}")
        lines.append(f"  mean users below safe floor: {power.get('mean_final_num_users_below_safe_floor')}")
        lines.append(f"  mean final sum rate bps: {power.get('mean_final_sum_rate_bps')}")
        lines.append(f"  mean final total power watt: {power.get('mean_final_total_power_watt')}")
        lines.append(f"  final avg serving APs/user: {power.get('final_avg_serving_aps_per_user')}")
        lines.append(f"  solver status counts: {power.get('solver_status_counts')}")
        lines.append("")

    lines.append("Recommended main plot keys")
    lines.append("-" * 80)
    lines.append(json.dumps(meta.get("recommended_main_plot_keys", {}), indent=2))
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Inspect mode
# -----------------------------------------------------------------------------


def inspect_run(args: argparse.Namespace) -> None:
    run_dir = build_run_dir(args)
    summary_path = run_dir / "merged_summary.txt"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
        return
    chunk_summaries = sorted((run_dir / "chunks").glob("summary_chunk_*.txt"))
    if not chunk_summaries:
        raise FileNotFoundError(f"No merged summary or chunk summaries found under {run_dir}")
    print(f"No merged summary yet. Found {len(chunk_summaries)} chunk summaries. Showing the latest:\n")
    print(chunk_summaries[-1].read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate convergence behavior data for RIS/cell-free optimizers.")

    p.add_argument("--mode", choices=("run_chunk", "merge", "inspect"), required=True)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--run-stamp", type=str, default=None)
    p.add_argument("--tag", type=str, default="convergence")

    p.add_argument("--total-setups", type=int, default=EXPERIMENT_TOTAL_SETUPS)
    p.add_argument("--total-chunks", type=int, default=EXPERIMENT_DEFAULT_TOTAL_CHUNKS)
    p.add_argument("--expected-chunks", type=int, default=None)
    p.add_argument("--chunk-id", type=int, default=None)
    p.add_argument("--base-seed", type=int, default=EXPERIMENT_BASE_SEED)
    p.add_argument("--config-overrides-json", type=str, default=None,
                   help="Optional JSON string/path with dot-path config overrides. Values here override EXPERIMENT_CONFIG_OVERRIDES.")

    # Shared pipeline options.
    p.add_argument("--pilot-scheme", choices=("heuristic", "random"), default="heuristic")
    p.add_argument("--q-neighbors", type=int, default=3)
    p.add_argument("--protect-masters", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--phase-mode", choices=("random", "equal"), default="equal")
    p.add_argument("--equal-phase-deg", type=float, default=45.0)

    # Association game options.
    p.add_argument("--assoc-d-th", type=float, default=100.0)
    p.add_argument("--assoc-init-action-mode", choices=("full_candidate", "random", "strongest_singleton"),
                   default="full_candidate")
    p.add_argument("--assoc-response-mode", choices=("best_response", "logit"),
                   default="best_response")
    p.add_argument("--assoc-omega-initial", type=float, default=4.0)
    p.add_argument("--assoc-max-game-iters", type=int, default=50)
    p.add_argument("--assoc-max-dinkelbach-iters", type=int, default=10)
    p.add_argument("--assoc-dinkelbach-tol", type=float, default=1e-3)
    p.add_argument("--assoc-convergence-window", type=int, default=3)
    p.add_argument("--assoc-stability-tol", type=float, default=1e-3)
    p.add_argument("--assoc-utility-improvement-tol", type=float, default=1e-12)
    p.add_argument("--assoc-max-subset-size", type=int, default=None)
    p.add_argument("--assoc-max-actions-per-user", type=int, default=None)
    p.add_argument("--assoc-incumbent-tracking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--assoc-outer-warm-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--assoc-cycle-detection", action=argparse.BooleanOptionalAction, default=True)

    # Power optimizer options.
    p.add_argument("--power-qos-target-mode", choices=("fixed", "scaled_equal_baseline"),
                   default="scaled_equal_baseline")
    p.add_argument("--power-qos-sinr-target", type=float, default=1e-6)
    p.add_argument("--power-qos-kappa", type=float, default=0.6)
    p.add_argument("--power-max-dinkelbach-iters", type=int, default=25)
    p.add_argument("--power-max-sca-iters", type=int, default=20)
    p.add_argument("--power-dinkelbach-tol", type=float, default=1e-3)
    p.add_argument("--power-sca-tol", type=float, default=1e-4)
    p.add_argument("--power-qos-tol", type=float, default=1e-6)
    p.add_argument("--power-solver", type=str, default="CLARABEL")
    p.add_argument("--power-fallback-solvers", type=str, default="SCS")
    p.add_argument("--power-solver-verbose", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--power-warm-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--power-safe-user-power-floor-watt", type=float, default=1e-6)

    # Debug logging controls. Keep off by default to avoid huge cluster logs.
    p.add_argument("--power-debug-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--power-debug-capture-solver-output", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--power-debug-check-qmat-psd", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--power-debug-zero-power-users", action=argparse.BooleanOptionalAction, default=True)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.expected_chunks is None:
        args.expected_chunks = args.total_chunks

    if args.mode == "run_chunk":
        run_chunk(args)
    elif args.mode == "merge":
        merge_chunks(args)
    elif args.mode == "inspect":
        inspect_run(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
