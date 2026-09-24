# ITNG YMYFA 80asra YA
# IMZZ

# src/power_allocation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import warnings

from .config import Config


PowerScheme = Literal["equal", "optimized"]



@dataclass(frozen=True)
class PowerAllocationSetups:
    """
    Downlink power allocation outputs for all setups.

    Shapes
    - eta: (S, L, K) in Watt
    - eta_sqrt: (S, L, K)
    - eta_safe: (S, L, K) or None
      Saved floor-safe accepted point for reporting. Setups with no saved safe
      point are filled with NaN when eta_safe is present.
    - num_served_users: (S, L)
    - ap_total_power: (S, L)

    Debug
    - optimized_debug_by_setup is None for non-optimized branches
    - optimized_debug_by_setup has length S for the optimized branch
    """
    eta: np.ndarray
    eta_sqrt: np.ndarray
    num_served_users: np.ndarray
    ap_total_power: np.ndarray
    serve_mask_used: np.ndarray
    scheme: PowerScheme
    pmax_watt_per_ap: float
    optimized_debug_by_setup: Optional[list["OptimizedPowerDebugSetup"]] = None
    eta_safe: np.ndarray | None = None



@dataclass(frozen=True)
class OptimizedPowerOptions:
    """
    Numerical options for the QoS-constrained EE optimizer.

    QoS modes
    - fixed
        Use the same SINR target for every user, i.e. Gamma_k = qos_sinr_target.
    - scaled_equal_baseline
        Use a user-dependent target based on equal-power initialization,
        i.e. Gamma_k = qos_kappa * gamma_k_eq.

    Notes
    - For scaled_equal_baseline, qos_kappa must lie in [0, 1].
    - If qos_kappa <= 1, the equal-power initialization is feasible by construction.
    """
    qos_target_mode: Literal["fixed", "scaled_equal_baseline"] = "scaled_equal_baseline"
    qos_sinr_target: float = 1e-6               # used when the mode is fixed
    qos_kappa: float = 0.4

    max_dinkelbach_iters: int = 25
    max_sca_iters: int = 20

    dinkelbach_tol: float = 1e-3                
    sca_tol: float = 1e-4
    qos_tol: float = 1e-2                       # Numerical tolerance for QoS feasibility checks

    solver: str = "CLARABEL"
    fallback_solvers: tuple[str, ...] = ("SCS",)
    solver_verbose: bool = False
    warm_start: bool = True

    psd_eig_floor: float = 1e-10                # Used when projecting quadratic matrices onto the PSD cone
    numeric_eps: float = 1e-12                  # Used to avoid division by zero

    validate_final_result: bool = True
    warn_on_final_qos_infeasible: bool = True

    accept_inaccurate_with_postcheck: bool = True           # Lets the code accept a solver point marked inaccurate only if the post-check says it is actually okay.
    inactive_link_tol: float = 1e-8
    ap_power_budget_tol: float = 1e-6
    candidate_objective_abs_tol: float = 1e-6
    candidate_objective_rel_tol: float = 1e-2

    # Debug controls
    debug_enabled: bool = True                      # Master switch for file logging.
    debug_log_path: Optional[str] = None
    debug_capture_solver_output: bool = False       # If enabled, captures solver terminal output into the debug file.
    debug_check_qmat_psd: bool = False              # Adds detailed PSD diagnostics for the Q-matrices before optimization.
    debug_zero_power_users: bool = True             # Logs user total powers before and after optimization, using the zero-power threshold below.

    # Thresholds for PSD diagnostics
    debug_psd_abs_tol: float = 1e-9
    debug_psd_rel_tol: float = 1e-12

    # Threshold for declaring a user's total allocated power effectively zero
    debug_zero_power_tol: float = 1e-6

    # Reporting threshold for a saved accepted point with all users above the floor
    safe_user_power_floor_watt: float = 1e-6



@dataclass(frozen=True)
class _OptimizedPowerCommonData:
    """
    Closed-form ingredients that are common to all setups.
    """
    p_vec: np.ndarray
    chi: np.ndarray
    iota: np.ndarray
    delta_all: np.ndarray
    a: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    c: np.ndarray
    R_f: np.ndarray
    W: np.ndarray
    beta_lk: np.ndarray
    hat_beta: np.ndarray
    tilde_beta: np.ndarray
    Xi: np.ndarray | None
    Gamma: np.ndarray | None
    pilot_groups: list[list[np.ndarray]]
    zeta_fn: Any



@dataclass(frozen=True)
class _SetupOptimizationContext:
    """
    One-setup constants for the optimized power solver.
    """
    setup_index: int
    serve_mask: np.ndarray
    delta_ap_active: np.ndarray
    user_serving_count: np.ndarray

    ds_coeff: np.ndarray
    q_mats: np.ndarray
    q_nonzero: np.ndarray

    prelog_factor: float
    bandwidth_hz: float
    pa_efficiency: float
    ap_circuit_power_watt: float
    ris_static_power_watt: float
    fronthaul_energy_per_bit_joule: float
    pmax_watt_per_ap: float
    eps: float



@dataclass(frozen=True)
class _SetupTrueMetrics:
    """
    True one-setup metrics evaluated from rho and the precomputed context.
    """
    rho: np.ndarray
    eta: np.ndarray
    sinr: np.ndarray
    se: np.ndarray
    psi: np.ndarray
    ds_power: np.ndarray

    sum_rate_bps: float
    total_power_watt: float
    ee_bit_per_joule: float

    tx_power_watt: float
    circuit_power_watt: float
    ris_power_watt: float
    fronthaul_power_watt: float



@dataclass(frozen=True)
class _CandidateAcceptanceCheck:
    """
    Acceptance result for one solver candidate.
    """
    finite_ok: bool
    budget_ok: bool
    inactive_ok: bool
    qos_ok: bool
    objective_ok: bool
    min_qos_margin: float
    transformed_objective: float
    allowed_objective_drop: float
    accepted: bool
    message: str



@dataclass(frozen=True)
class OptimizedPowerDebugSetup:
    """
    Public debug container for one optimized-power setup.
    """
    initial_feasible: bool
    used_equal_power_init: bool

    alpha_history: list[float]
    dinkelbach_residual_history: list[float]
    sca_rho_change_history: list[float]
    min_qos_margin_history: list[float]
    objective_history: list[float]
    min_user_total_power_history: list[float]
    num_users_below_safe_floor_history: list[int]

    solver_status_history: list[str]
    solver_name_history: list[str]

    converged_dinkelbach: bool
    converged_sca_last_outer: bool
    num_dinkelbach_iters: int
    num_sca_iters_last_outer: int

    init_user_total_power: list[float]
    final_user_total_power: list[float]
    zero_power_users_before: list[int]
    zero_power_users_after: list[int]

    qos_target: list[float]
    safe_user_power_floor_watt: float
    safe_point_found: bool
    safe_point_outer_iter: Optional[int]
    safe_point_sca_iter: Optional[int]
    safe_point_user_total_power: Optional[list[float]]
    safe_point_ee_bit_per_joule: Optional[float]
    safe_point_sum_rate_bps: Optional[float]
    safe_point_total_power_watt: Optional[float]
    safe_point_min_qos_margin: Optional[float]
    safe_point_transformed_objective: Optional[float]
    safe_point_min_user_total_power: Optional[float]
    safe_point_num_users_below_safe_floor: Optional[int]

    final_qos_feasible: bool
    final_min_qos_margin: float
    final_min_user_total_power: float
    final_num_users_below_safe_floor: int
    final_warning_message: Optional[str]

    stopped_on_rejected_candidate: bool
    stop_reason: str



def _validate_or_build_serve_mask(
    cfg: Config,
    serve_mask: Optional[np.ndarray],
    serve_users: Optional[Sequence[Sequence[Sequence[int]]]],
) -> np.ndarray:
    """
    Return a boolean serving mask with shape (S, L, K).

    Accepts either serve_mask or serve_users.
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    if serve_mask is not None and serve_users is not None:
        raise ValueError("Provide either serve_mask or serve_users, not both")

    if serve_mask is not None:
        m = np.asarray(serve_mask)
        if m.shape != (S, L, K):
            raise ValueError("serve_mask must have shape (S, L, K)")
        return m.astype(bool, copy=False)

    if serve_users is None:
        raise ValueError("You must provide serve_mask or serve_users")

    first_entry_is_sequence = False
    if S == 1 and len(serve_users) == L:
        first_entry_is_sequence = (
            len(serve_users) > 0
            and len(serve_users[0]) > 0
            and isinstance(serve_users[0][0], (list, tuple, np.ndarray))
        )
    if S == 1 and len(serve_users) == L and not first_entry_is_sequence:
        raise ValueError("serve_users format is ambiguous for S == 1. Use serve_users[0][l] = [...]")

    if len(serve_users) == L and S == 1:
        serve_users = [serve_users]  # type: ignore[assignment]

    if len(serve_users) != S:
        raise ValueError("serve_users must have length S, where serve_users[s] has length L")

    mask = np.zeros((S, L, K), dtype=bool)

    for s in range(S):
        if len(serve_users[s]) != L:
            raise ValueError("serve_users[s] must have length L")
        for l in range(L):
            users_l = np.asarray(serve_users[s][l], dtype=int).ravel()
            if users_l.size == 0:
                continue
            if np.any(users_l < 0) or np.any(users_l >= K):
                raise ValueError("serve_users contains user indices out of range")
            mask[s, l, users_l] = True

    return mask



def _allocate_downlink_power_equal(
    cfg: Config,
    m: np.ndarray,
) -> PowerAllocationSetups:
    """
    Existing equal-power rule kept exactly as simple as before.
    """
    Pmax = float(cfg.power.dl_max_power_watt_per_ap)
    num_served = m.sum(axis=2).astype(int)

    denom = np.maximum(num_served.astype(float), 1.0)
    eta = (Pmax / denom)[:, :, None] * m.astype(float)

    ap_total_power = eta.sum(axis=2)
    if np.any(ap_total_power > Pmax + 1e-9):
        raise RuntimeError("Power budget violated in equal scheme.")

    eta_sqrt = np.where(eta > 0.0, np.sqrt(eta), 0.0)

    return PowerAllocationSetups(
        eta=eta,
        eta_sqrt=eta_sqrt,
        num_served_users=num_served,
        ap_total_power=ap_total_power,
        serve_mask_used=m,
        scheme="equal",
        pmax_watt_per_ap=Pmax,
        optimized_debug_by_setup=None,
    )



def _symmetrize_real_matrix(M: np.ndarray) -> np.ndarray:
    """
    Convert a possibly complex or slightly asymmetric matrix into a real symmetric one.
    """
    M = np.asarray(M)
    M_real = np.real(M)
    return 0.5 * (M_real + M_real.T)



def _matrix_symmetry_error(M: np.ndarray) -> float:
    """
    Frobenius norm of the asymmetric part of the real matrix.
    """
    M = np.asarray(M)
    M_real = np.real(M)
    return float(np.linalg.norm(M_real - M_real.T, ord="fro"))



def _real_quadratic_coeff(value: complex) -> float:
    """
    For Q-matrix assembly, keep only the real part of a theoretically real
    quadratic-form coefficient.
    Do not clip elementwise to be non-negative. PSD projection handles that
    at the matrix level.
    """
    real_val = float(np.real(complex(value)))
    if not np.isfinite(real_val):
        raise RuntimeError(f"Non-finite quadratic coefficient encountered: {value}")
    return real_val



def _project_psd(M: np.ndarray, eig_floor: float) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone.

    The quadratic forms are theoretically PSD. In finite precision, especially
    in the RIS case with very large matrix magnitudes, tiny negative eigenvalues
    can remain even after a single projection. This routine keeps projecting and
    applies a minimal diagonal shift when needed so that the returned matrix is
    numerically PSD for CVXPY.
    """
    M_psd = _symmetrize_real_matrix(M)
    floor = max(float(eig_floor), 0.0)
    eye = np.eye(M_psd.shape[0], dtype=float)

    for _ in range(3):
        eigvals, eigvecs = np.linalg.eigh(M_psd)
        eigvals = np.maximum(eigvals, 0.0)
        M_psd = (eigvecs * eigvals[None, :]) @ eigvecs.T
        M_psd = _symmetrize_real_matrix(M_psd)

        min_after = float(np.min(np.linalg.eigvalsh(M_psd)))
        if min_after >= 0.0:
            return M_psd

        M_psd = _symmetrize_real_matrix(
            M_psd + (-min_after + floor) * eye
        )

    min_after = float(np.min(np.linalg.eigvalsh(M_psd)))
    if min_after < 0.0:
        M_psd = _symmetrize_real_matrix(
            M_psd + (-min_after + floor) * eye
        )

    return _symmetrize_real_matrix(M_psd)



def _build_common_closed_form_data(
    cfg: Config,
    est: Any,
    corr: Any,
    ls: Any,
    pilot_of_user: np.ndarray,
    phases: Any,
    pilot_power_watt_per_user: Optional[np.ndarray],
) -> _OptimizedPowerCommonData:
    """
    Build the setup-independent closed-form ingredients once.
    """
    from .dl_sinr_closed_form import (
        _build_a_b_mu_c,
        _compute_chi_iota_per_setup,
        _get_delta_all,
        _pilot_groups_from_indices,
        _zeta_lk_A,
    )

    S = int(cfg.sim.num_setups)
    K = int(cfg.dims.num_users)
    tau_p = int(round(float(cfg.tau_p())))

    pilot_of_user = np.asarray(pilot_of_user, dtype=int)
    if pilot_of_user.shape != (S, K):
        raise ValueError("pilot_of_user must have shape (S, K)")

    if pilot_power_watt_per_user is None:
        p_vec = np.full(K, float(cfg.pilots.pilot_power_watt), dtype=float)
    else:
        p_vec = np.asarray(pilot_power_watt_per_user, dtype=float).reshape(-1)
        if p_vec.shape[0] != K:
            raise ValueError("pilot_power_watt_per_user must have length K")
        if np.any(p_vec < 0):
            raise ValueError("pilot_power_watt_per_user must be non-negative")

    chi, iota = _compute_chi_iota_per_setup(cfg, corr, phases)
    delta_all = _get_delta_all(cfg, corr)
    a, b, mu, c = _build_a_b_mu_c(est, delta_all)

    beta_lk = np.asarray(ls.ap_user_beta_over_noise, dtype=float)
    R_f = np.asarray(corr.R_ap_user, dtype=np.complex128)
    W = np.asarray(est.W, dtype=np.complex128)

    T = 0 if corr.delta_ap_ris.ndim < 3 else int(corr.delta_ap_ris.shape[2])
    if cfg.ris.enable_ris and T > 0:
        hat_beta = np.asarray(ls.ap_ris_beta_over_noise[:, :, 0], dtype=float)
        tilde_beta = np.asarray(ls.ris_user_beta_over_noise[:, 0, :], dtype=float)
        R_ris = np.asarray(corr.R_ris, dtype=np.complex128)

        Xi = np.zeros((S, K, R_ris.shape[0], R_ris.shape[1]), dtype=np.complex128)
        Gamma = np.zeros((S, int(cfg.dims.num_aps), R_ris.shape[0], R_ris.shape[1]), dtype=np.complex128)

        vartheta = np.asarray(phases.vartheta, dtype=float)
        for s in range(S):
            d = np.exp(1j * vartheta[s])
            Theta_R_ThetaH = (d[:, None] * R_ris) * np.conj(d[None, :])
            for k in range(K):
                Xi[s, k] = tilde_beta[s, k] * Theta_R_ThetaH
            for l in range(int(cfg.dims.num_aps)):
                Gamma[s, l] = hat_beta[s, l] * R_ris
    else:
        hat_beta = np.zeros((S, int(cfg.dims.num_aps)), dtype=float)
        tilde_beta = np.zeros((S, K), dtype=float)
        Xi = None
        Gamma = None

    pilot_groups = _pilot_groups_from_indices(pilot_of_user, tau_p)

    return _OptimizedPowerCommonData(
        p_vec=p_vec,
        chi=chi,
        iota=iota,
        delta_all=delta_all,
        a=a,
        b=b,
        mu=mu,
        c=c,
        R_f=R_f,
        W=W,
        beta_lk=beta_lk,
        hat_beta=hat_beta,
        tilde_beta=tilde_beta,
        Xi=Xi,
        Gamma=Gamma,
        pilot_groups=pilot_groups,
        zeta_fn=_zeta_lk_A,
    )



def _build_qos_target_vector(
    eq_sinr: np.ndarray,
    options: OptimizedPowerOptions,
) -> np.ndarray:
    """
    Build the per-user SINR target vector Gamma_k.
    """
    eq_sinr = np.maximum(np.asarray(eq_sinr, dtype=float), 0.0)

    if options.qos_target_mode == "fixed":
        gamma_qos = np.full(eq_sinr.shape, float(options.qos_sinr_target), dtype=float)

    elif options.qos_target_mode == "scaled_equal_baseline":
        kappa = float(options.qos_kappa)
        if not (0.0 <= kappa <= 1.0):
            raise ValueError("qos_kappa must lie in [0, 1].")
        gamma_qos = kappa * eq_sinr

    else:
        raise ValueError(f"Unknown qos_target_mode {options.qos_target_mode!r}")

    gamma_qos = np.maximum(gamma_qos, 0.0)

    if np.any(~np.isfinite(gamma_qos)):
        raise RuntimeError("Non-finite QoS target encountered.")

    return gamma_qos



def _debug_log(options: OptimizedPowerOptions, msg: str) -> None:
    """
    Append a debug message to the configured log file when debugging is enabled.
    """
    if not options.debug_enabled or options.debug_log_path is None:
        return

    log_path = Path(options.debug_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")



def _derive_report_log_path(debug_log_path: str) -> Path:
    """
    Build a report path in the same directory as the debug log.

    If the debug file is named like debug_TAG.txt or power_debug_TAG.txt,
    the report file becomes report_TAG.txt. Otherwise we use the whole stem.
    """
    log_path = Path(debug_log_path)
    stem = log_path.stem

    if stem.startswith("power_debug_"):
        tag = stem[len("power_debug_"):]
    elif stem.startswith("debug_"):
        tag = stem[len("debug_"):]
    else:
        tag = stem

    return log_path.with_name(f"report_{tag}.txt")



def _report_log(options: OptimizedPowerOptions, msg: str) -> None:
    """
    Append a compact report-style message to the derived report log file.
    """
    if not options.debug_enabled or options.debug_log_path is None:
        return

    report_path = _derive_report_log_path(options.debug_log_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")



def _reset_report_log(options: OptimizedPowerOptions) -> None:
    """
    Clear the report log at the start of a new optimized-power run.
    """
    if not options.debug_enabled or options.debug_log_path is None:
        return

    report_path = _derive_report_log_path(options.debug_log_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("", encoding="utf-8")



def _brief_point_summary(
    metrics: _SetupTrueMetrics,
    gamma_qos: np.ndarray,
    user_total_power: np.ndarray,
    safe_floor: float,
) -> str:
    """
    Build a compact one-line summary for a point.
    """
    user_total_power = np.asarray(user_total_power, dtype=float)
    below_floor_users = np.where(user_total_power <= float(safe_floor))[0].astype(int).tolist()
    min_user_power = float(np.min(user_total_power)) if user_total_power.size > 0 else 0.0
    min_qos_margin = float(np.min(metrics.sinr - np.asarray(gamma_qos, dtype=float)))

    return (
        f"ee={metrics.ee_bit_per_joule:.6e} "
        f"sum_rate={metrics.sum_rate_bps:.6e} "
        f"total_power={metrics.total_power_watt:.6e} "
        f"min_qos_margin={min_qos_margin:.3e} "
        f"min_user_power={min_user_power:.6e} "
        f"users_below_floor={below_floor_users}"
    )



def _matrix_psd_diagnostics(M: np.ndarray) -> tuple[float, float, float]:
    """
    Return PSD diagnostics for a real symmetric version of M.
    """
    M_sym = _symmetrize_real_matrix(M)
    eigvals = np.linalg.eigvalsh(M_sym)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    fro_norm = float(np.linalg.norm(M_sym, ord="fro"))
    return min_eig, max_eig, fro_norm



def _is_effectively_non_psd(
    min_eig: float,
    fro_norm: float,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """
    Decide whether a negative eigenvalue is large enough to matter.
    """
    thresh = max(abs_tol, rel_tol * max(1.0, fro_norm))
    return bool(min_eig < -thresh)



def _user_total_power_report(
    eta: np.ndarray,
    zero_tol: float,
) -> tuple[np.ndarray, list[int]]:
    """
    Return per-user total power and effectively zero-power users for one setup.
    """
    user_total = np.sum(np.asarray(eta, dtype=float), axis=0)
    zero_users = np.where(user_total <= float(zero_tol))[0].astype(int).tolist()
    return user_total, zero_users



def _power_floor_status(
    eta: np.ndarray,
    safe_floor: float,
) -> tuple[np.ndarray, list[int], float, bool]:
    """
    Return per-user total power, users at or below the safe floor, the minimum user
    total power, and whether all users are strictly above the safe floor.
    """
    user_total = np.sum(np.asarray(eta, dtype=float), axis=0)
    below_floor_users = np.where(user_total <= float(safe_floor))[0].astype(int).tolist()
    min_user_power = float(np.min(user_total)) if user_total.size > 0 else 0.0
    all_above_floor = bool(len(below_floor_users) == 0)
    return user_total, below_floor_users, min_user_power, all_above_floor



def _log_qmat_debug_entry(
    *,
    options: OptimizedPowerOptions,
    setup_index: int,
    ris_enabled: bool,
    target_user: int,
    source_user: int,
    block: str,
    same_pilot: Optional[bool],
    raw_matrix: np.ndarray,
    proj_matrix: np.ndarray,
) -> tuple[bool, bool, float, float]:
    """
    Log one Q-matrix diagnostic entry before CVXPY sees the matrix.

    Returns
    - raw_is_bad
    - proj_is_bad
    - raw_min_eig
    - proj_min_eig
    """
    raw_min_eig, raw_max_eig, raw_fro = _matrix_psd_diagnostics(raw_matrix)
    proj_min_eig, proj_max_eig, proj_fro = _matrix_psd_diagnostics(proj_matrix)

    raw_sym_err = _matrix_symmetry_error(raw_matrix)
    proj_sym_err = _matrix_symmetry_error(proj_matrix)
    raw_max_abs = float(np.max(np.abs(np.real(raw_matrix)))) if raw_matrix.size else 0.0
    proj_max_abs = float(np.max(np.abs(np.real(proj_matrix)))) if proj_matrix.size else 0.0

    raw_is_bad = _is_effectively_non_psd(
        raw_min_eig,
        raw_fro,
        options.debug_psd_abs_tol,
        options.debug_psd_rel_tol,
    )
    proj_is_bad = _is_effectively_non_psd(
        proj_min_eig,
        proj_fro,
        options.debug_psd_abs_tol,
        options.debug_psd_rel_tol,
    )

    label = "[QMAT PSD WARNING]" if (raw_is_bad or proj_is_bad) else "[QMAT PSD CHECK]"
    same_pilot_str = "" if same_pilot is None else f" same_pilot={int(same_pilot)}"

    _debug_log(
        options,
        f"{label} "
        f"RIS={int(ris_enabled)} setup={setup_index} target_user={target_user} source_user={source_user} block={block}"
        f"{same_pilot_str} "
        f"raw_min_eig={raw_min_eig:.3e} raw_max_eig={raw_max_eig:.3e} raw_fro={raw_fro:.3e} raw_sym_err={raw_sym_err:.3e} raw_max_abs={raw_max_abs:.3e} "
        f"proj_min_eig={proj_min_eig:.3e} proj_max_eig={proj_max_eig:.3e} proj_fro={proj_fro:.3e} proj_sym_err={proj_sym_err:.3e} proj_max_abs={proj_max_abs:.3e}"
    )

    return raw_is_bad, proj_is_bad, raw_min_eig, proj_min_eig



def _build_setup_optimization_context(
    cfg: Config,
    setup_index: int,
    serve_mask_s: np.ndarray,
    pilot_of_user_s: np.ndarray,
    common: _OptimizedPowerCommonData,
    options: OptimizedPowerOptions,
) -> _SetupOptimizationContext:
    """
    Build the one-setup constants needed by the SCA + Dinkelbach optimizer.
    """
    s = int(setup_index)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    tau_p = int(round(float(cfg.tau_p())))
    tau_c = float(cfg.pilots.coherence_block_length)
    tau_d = tau_c - float(tau_p)
    if tau_d < 0:
        raise ValueError("tau_d must be non-negative")

    eps = float(options.numeric_eps)

    serve_mask_s = np.asarray(serve_mask_s, dtype=bool)
    if serve_mask_s.shape != (L, K):
        raise ValueError("serve_mask_s must have shape (L, K)")

    delta_ap_active = np.any(serve_mask_s, axis=1).astype(float)
    user_serving_count = np.sum(serve_mask_s, axis=0).astype(int)

    mu_s = np.asarray(common.mu[s], dtype=float)
    mu_safe = np.maximum(mu_s, eps)
    ds_coeff = np.where(serve_mask_s, np.sqrt(mu_safe), 0.0)

    a_s = np.asarray(common.a[s], dtype=np.complex128)
    b_s = np.asarray(common.b[s], dtype=np.complex128)
    c_s = np.asarray(common.c[s], dtype=np.complex128)
    beta_s = np.asarray(common.beta_lk[s], dtype=float)
    hat_beta_s = np.asarray(common.hat_beta[s], dtype=float)
    tilde_beta_s = np.asarray(common.tilde_beta[s], dtype=float)
    chi_s = float(common.chi[s])
    iota_s = float(common.iota[s])
    p_vec = np.asarray(common.p_vec, dtype=float)

    q_mats = np.zeros((K, K, L, L), dtype=float)
    q_nonzero = np.zeros((K, K), dtype=bool)

    raw_bad_count = 0
    proj_bad_count = 0
    worst_raw_min = np.inf
    worst_proj_min = np.inf

    for k in range(K):
        pilot_k = int(pilot_of_user_s[k])
        users_same_pilot_k = np.asarray(common.pilot_groups[s][pilot_k], dtype=int)

        M_uu = np.zeros((L, L), dtype=float)
        serving_k = np.where(serve_mask_s[:, k])[0].astype(int)

        for l in serving_k:
            for m in serving_k:
                if m == l:
                    continue

                term1 = (
                    p_vec[k] * tau_p
                    * (
                        beta_s[l, k] * a_s[l, k]
                        + hat_beta_s[l] * tilde_beta_s[k] * a_s[l, k] * chi_s
                    )
                    * np.conj(
                        beta_s[m, k] * a_s[m, k]
                        + hat_beta_s[m] * tilde_beta_s[k] * a_s[m, k] * chi_s
                    )
                )

                same_pilot_sum = 0.0 + 0.0j
                for i in users_same_pilot_k:
                    same_pilot_sum += (
                        p_vec[i] * tau_p
                        * hat_beta_s[l] * hat_beta_s[m]
                        * tilde_beta_s[k] * tilde_beta_s[i]
                        * a_s[l, k] * np.conj(a_s[m, k]) * iota_s
                    )

                coeff_lm = (term1 + same_pilot_sum - mu_s[l, k] * mu_s[m, k]) / np.sqrt(mu_safe[l, k] * mu_safe[m, k])
                M_uu[l, m] += _real_quadratic_coeff(coeff_lm)

        for l in serving_k:
            if cfg.ris.enable_ris and common.Xi is not None and common.Gamma is not None:
                zeta_val = max(
                    0.0,
                    _real_quadratic_coeff(
                        common.zeta_fn(
                            R_f_lk=common.R_f[s, l, k],
                            Delta=common.delta_all[s, l],
                            A=common.W[s, l, k],
                            hat_beta_l=hat_beta_s[l],
                            tilde_beta_k=tilde_beta_s[k],
                            chi=chi_s,
                            iota=iota_s,
                            Gamma_l=common.Gamma[s, l],
                            Xi_k=common.Xi[s, k],
                        )
                    ),
                )
            else:
                A = common.W[s, l, k]
                A_H = A.conj().T
                zeta_raw = (
                    np.trace(common.R_f[s, l, k] @ A) * np.trace(common.R_f[s, l, k] @ A_H)
                    + np.trace(common.R_f[s, l, k] @ A @ common.R_f[s, l, k] @ A_H)
                )
                zeta_val = max(0.0, _real_quadratic_coeff(zeta_raw))

            diag_term = (
                p_vec[k] * tau_p * zeta_val
                + c_s[l, k, k] * 1.0
                - (mu_s[l, k] ** 2)
            )

            extra_sum = 0.0 + 0.0j
            for i in users_same_pilot_k:
                if i == k:
                    continue
                extra_sum += (
                    p_vec[i] * tau_p
                    * (
                        beta_s[l, k] * hat_beta_s[l] * tilde_beta_s[i] * b_s[l, k] * chi_s
                        + beta_s[l, k] * beta_s[l, i] * b_s[l, k]
                        + (hat_beta_s[l] ** 2) * tilde_beta_s[k] * tilde_beta_s[i]
                        * (np.abs(a_s[l, k]) ** 2 * iota_s + b_s[l, k] * (chi_s ** 2))
                    )
                )

            coeff_ll = (diag_term + extra_sum) / mu_safe[l, k]
            M_uu[l, l] += _real_quadratic_coeff(coeff_ll)

        Q_uu = _project_psd(M_uu, eig_floor=float(options.psd_eig_floor))

        if options.debug_enabled and options.debug_check_qmat_psd:
            raw_bad, proj_bad, raw_min, proj_min = _log_qmat_debug_entry(
                options=options,
                setup_index=s,
                ris_enabled=bool(cfg.ris.enable_ris),
                target_user=k,
                source_user=k,
                block="UU",
                same_pilot=None,
                raw_matrix=M_uu,
                proj_matrix=Q_uu,
            )
            raw_bad_count += int(raw_bad)
            proj_bad_count += int(proj_bad)
            worst_raw_min = min(worst_raw_min, raw_min)
            worst_proj_min = min(worst_proj_min, proj_min)

        q_mats[k, k] = Q_uu
        q_nonzero[k, k] = bool(np.linalg.norm(Q_uu, ord="fro") > 0.0)

        for kp in range(K):
            if kp == k:
                continue

            serving_kp = np.where(serve_mask_s[:, kp])[0].astype(int)
            if serving_kp.size == 0:
                continue

            pilot_kp = int(pilot_of_user_s[kp])
            users_same_pilot_kp = np.asarray(common.pilot_groups[s][pilot_kp], dtype=int)
            same_pilot = (pilot_kp == pilot_k)

            M_int = np.zeros((L, L), dtype=float)

            for l in serving_kp:
                for m in serving_kp:
                    if m == l:
                        continue

                    if same_pilot:
                        term1 = (
                            p_vec[k] * tau_p
                            * (
                                beta_s[l, k] * a_s[l, kp]
                                + hat_beta_s[l] * tilde_beta_s[k] * a_s[l, kp] * chi_s
                            )
                            * np.conj(
                                beta_s[m, k] * a_s[m, kp]
                                + hat_beta_s[m] * tilde_beta_s[k] * a_s[m, kp] * chi_s
                            )
                        )
                    else:
                        term1 = 0.0 + 0.0j

                    same_pilot_sum = 0.0 + 0.0j
                    for i in users_same_pilot_kp:
                        same_pilot_sum += (
                            p_vec[i] * tau_p
                            * hat_beta_s[l] * hat_beta_s[m]
                            * tilde_beta_s[k] * tilde_beta_s[i]
                            * a_s[l, kp] * np.conj(a_s[m, kp]) * iota_s
                        )

                    coeff_lm = (term1 + same_pilot_sum) / np.sqrt(mu_safe[l, kp] * mu_safe[m, kp])
                    M_int[l, m] += _real_quadratic_coeff(coeff_lm)

            for l in serving_kp:
                if cfg.ris.enable_ris and common.Xi is not None and common.Gamma is not None:
                    zeta_val = common.zeta_fn(
                        R_f_lk=common.R_f[s, l, k],
                        Delta=common.delta_all[s, l],
                        A=common.W[s, l, kp],
                        hat_beta_l=hat_beta_s[l],
                        tilde_beta_k=tilde_beta_s[k],
                        chi=chi_s,
                        iota=iota_s,
                        Gamma_l=common.Gamma[s, l],
                        Xi_k=common.Xi[s, k],
                    )
                    zeta_val = max(0.0, _real_quadratic_coeff(zeta_val))
                else:
                    A = common.W[s, l, kp]
                    A_H = A.conj().T
                    zeta_raw = (
                    np.trace(common.R_f[s, l, k] @ A) * np.trace(common.R_f[s, l, k] @ A_H)
                    + np.trace(common.R_f[s, l, k] @ A @ common.R_f[s, l, k] @ A_H)
                    )
                    zeta_val = max(0.0, _real_quadratic_coeff(zeta_raw))

                diag_term = c_s[l, kp, k] * 1.0
                if same_pilot:
                    diag_term += p_vec[k] * tau_p * zeta_val

                extra_sum = 0.0 + 0.0j
                for i in users_same_pilot_kp:
                    if same_pilot and i == k:
                        continue
                    extra_sum += (
                        p_vec[i] * tau_p
                        * (
                            beta_s[l, k] * hat_beta_s[l] * tilde_beta_s[i] * b_s[l, kp] * chi_s
                            + beta_s[l, i] * hat_beta_s[l] * tilde_beta_s[k] * b_s[l, kp] * chi_s
                            + beta_s[l, k] * beta_s[l, i] * b_s[l, kp]
                            + (hat_beta_s[l] ** 2) * tilde_beta_s[k] * tilde_beta_s[i]
                            * (np.abs(a_s[l, kp]) ** 2 * iota_s + b_s[l, kp] * (chi_s ** 2))
                        )
                    )

                coeff_ll = (diag_term + extra_sum) / mu_safe[l, kp]
                M_int[l, l] += _real_quadratic_coeff(coeff_ll)

            Q_int = _project_psd(M_int, eig_floor=float(options.psd_eig_floor))

            if options.debug_enabled and options.debug_check_qmat_psd:
                raw_bad, proj_bad, raw_min, proj_min = _log_qmat_debug_entry(
                    options=options,
                    setup_index=s,
                    ris_enabled=bool(cfg.ris.enable_ris),
                    target_user=k,
                    source_user=kp,
                    block="INT",
                    same_pilot=same_pilot,
                    raw_matrix=M_int,
                    proj_matrix=Q_int,
                )
                raw_bad_count += int(raw_bad)
                proj_bad_count += int(proj_bad)
                worst_raw_min = min(worst_raw_min, raw_min)
                worst_proj_min = min(worst_proj_min, proj_min)

            q_mats[k, kp] = Q_int
            q_nonzero[k, kp] = bool(np.linalg.norm(Q_int, ord="fro") > 0.0)

    if options.debug_enabled and options.debug_check_qmat_psd:
        if not np.isfinite(worst_raw_min):
            worst_raw_min = 0.0
        if not np.isfinite(worst_proj_min):
            worst_proj_min = 0.0
        _debug_log(
            options,
            "[QMAT SUMMARY] "
            f"RIS={int(cfg.ris.enable_ris)} setup={s} "
            f"raw_bad_blocks={raw_bad_count} proj_bad_blocks={proj_bad_count} "
            f"worst_raw_min_eig={worst_raw_min:.3e} worst_proj_min_eig={worst_proj_min:.3e}"
        )

    prelog_factor = float(tau_d) / float(tau_c)
    ris_power = (
        float(cfg.ee.ris_static_power_watt)
        if (cfg.ris.enable_ris and cfg.dims.num_ris > 0 and cfg.ee.use_ris_power_only_when_enabled)
        else (0.0 if cfg.ee.use_ris_power_only_when_enabled else float(cfg.ee.ris_static_power_watt))
    )

    return _SetupOptimizationContext(
        setup_index=s,
        serve_mask=serve_mask_s,
        delta_ap_active=delta_ap_active,
        user_serving_count=user_serving_count,
        ds_coeff=ds_coeff,
        q_mats=q_mats,
        q_nonzero=q_nonzero,
        prelog_factor=prelog_factor,
        bandwidth_hz=float(cfg.noise.bandwidth_hz),
        pa_efficiency=float(cfg.ee.pa_efficiency),
        ap_circuit_power_watt=float(cfg.ee.ap_circuit_power_watt),
        ris_static_power_watt=float(ris_power),
        fronthaul_energy_per_bit_joule=float(cfg.ee.fronthaul_energy_per_bit_joule),
        pmax_watt_per_ap=float(cfg.power.dl_max_power_watt_per_ap),
        eps=eps,
    )



def _evaluate_setup_true_metrics(
    ctx: _SetupOptimizationContext,
    rho: np.ndarray,
) -> _SetupTrueMetrics:
    """
    Evaluate the true one-setup SINR, SE, rate, power, and EE from rho.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.shape != ctx.serve_mask.shape:
        raise ValueError("rho must have shape (L, K)")

    rho = np.maximum(rho, 0.0)
    rho = rho * ctx.serve_mask.astype(float)
    eta = rho ** 2

    d_vec = np.sum(ctx.ds_coeff * rho, axis=0)
    ds_power = d_vec ** 2

    K = rho.shape[1]
    psi = np.ones(K, dtype=float)
    for k in range(K):
        val = 1.0
        for u in range(K):
            if ctx.q_nonzero[k, u]:
                ru = rho[:, u]
                val += float(ru.T @ ctx.q_mats[k, u] @ ru)
        psi[k] = max(val, ctx.eps)

    sinr = ds_power / psi
    sinr = np.maximum(sinr, 0.0)
    se = ctx.prelog_factor * np.log2(1.0 + sinr)

    sum_rate_bps = float(ctx.bandwidth_hz * np.sum(se))

    ap_eta_sum = np.sum(eta, axis=1)
    tx_power_watt = float((ctx.prelog_factor / ctx.pa_efficiency) * np.sum(ctx.delta_ap_active * ap_eta_sum))
    circuit_power_watt = float(ctx.ap_circuit_power_watt * np.sum(ctx.delta_ap_active))
    ris_power_watt = float(ctx.ris_static_power_watt)
    fronthaul_power_watt = float(
        ctx.bandwidth_hz * ctx.fronthaul_energy_per_bit_joule * np.sum(ctx.user_serving_count * se)
    )

    total_power_watt = tx_power_watt + circuit_power_watt + ris_power_watt + fronthaul_power_watt
    ee_bit_per_joule = sum_rate_bps / max(total_power_watt, ctx.eps)

    return _SetupTrueMetrics(
        rho=rho,
        eta=eta,
        sinr=sinr,
        se=se,
        psi=psi,
        ds_power=ds_power,
        sum_rate_bps=sum_rate_bps,
        total_power_watt=total_power_watt,
        ee_bit_per_joule=ee_bit_per_joule,
        tx_power_watt=tx_power_watt,
        circuit_power_watt=circuit_power_watt,
        ris_power_watt=ris_power_watt,
        fronthaul_power_watt=fronthaul_power_watt,
    )



def _validate_optimized_result_one_setup(
    ctx: _SetupOptimizationContext,
    metrics: _SetupTrueMetrics,
    gamma_qos: np.ndarray,
    options: OptimizedPowerOptions,
) -> tuple[bool, float]:
    """
    Lightweight internal validation helper.

    Hard numerical/model violations still raise.
    QoS infeasibility is returned as a boolean so the simulation can continue.
    """
    rho = metrics.rho
    eta = metrics.eta
    gamma_qos = np.asarray(gamma_qos, dtype=float)
    tol = max(float(options.qos_tol), 1e-10)

    if gamma_qos.shape != metrics.sinr.shape:
        raise ValueError("gamma_qos must have shape (K,)")

    if np.any(~np.isfinite(rho)) or np.any(~np.isfinite(eta)):
        raise RuntimeError("Optimized power allocation produced NaN or Inf values.")

    if np.any(rho < -tol) or np.any(eta < -tol):
        raise RuntimeError("Optimized power allocation produced negative power values.")

    inactive_eta = eta[~ctx.serve_mask]
    if inactive_eta.size > 0 and np.max(np.abs(inactive_eta)) > float(options.inactive_link_tol):
        raise RuntimeError("Inactive links received nonzero optimized power.")

    ap_total = np.sum(eta, axis=1)
    if np.any(ap_total > ctx.pmax_watt_per_ap + float(options.ap_power_budget_tol)):
        raise RuntimeError("Optimized power allocation violates the per-AP power budget.")

    qos_margin = metrics.sinr - gamma_qos
    min_qos_margin = float(np.min(qos_margin))
    qos_feasible = bool(np.all(qos_margin >= -tol))

    return qos_feasible, min_qos_margin



def _transformed_objective_value(
    metrics: _SetupTrueMetrics,
    alpha: float,
) -> float:
    """
    Return the Dinkelbach transformed objective evaluated at a true point.
    """
    return float(metrics.sum_rate_bps - alpha * metrics.total_power_watt)



def _evaluate_candidate_acceptance(
    ctx: _SetupOptimizationContext,
    metrics: _SetupTrueMetrics,
    gamma_qos: np.ndarray,
    alpha: float,
    reference_transformed_objective: float,
    options: OptimizedPowerOptions,
    *,
    require_objective_guard: bool,
) -> _CandidateAcceptanceCheck:
    """
    Post-check a returned solver point before accepting it.
    """
    tol = max(float(options.qos_tol), 1e-10)

    rho = metrics.rho
    eta = metrics.eta
    sinr = metrics.sinr
    se = metrics.se

    finite_ok = bool(
        np.all(np.isfinite(rho))
        and np.all(np.isfinite(eta))
        and np.all(np.isfinite(sinr))
        and np.all(np.isfinite(se))
        and np.all(np.isfinite(metrics.psi))
        and np.all(np.isfinite(metrics.ds_power))
        and np.isfinite(metrics.sum_rate_bps)
        and np.isfinite(metrics.total_power_watt)
        and np.isfinite(metrics.ee_bit_per_joule)
    )

    ap_total = np.sum(eta, axis=1)
    budget_ok = bool(
        np.all(ap_total <= ctx.pmax_watt_per_ap + float(options.ap_power_budget_tol))
    )

    inactive_eta = eta[~ctx.serve_mask]
    inactive_ok = bool(
        inactive_eta.size == 0
        or np.max(np.abs(inactive_eta)) <= float(options.inactive_link_tol)
    )

    min_qos_margin = float(np.min(sinr - gamma_qos))
    qos_ok = bool(min_qos_margin >= -tol)

    transformed_objective = _transformed_objective_value(metrics, alpha)
    allowed_objective_drop = max(
        float(options.candidate_objective_abs_tol),
        float(options.candidate_objective_rel_tol) * max(1.0, abs(reference_transformed_objective)),
    )

    if require_objective_guard:
        objective_ok = bool(
            transformed_objective >= reference_transformed_objective - allowed_objective_drop
        )
    else:
        objective_ok = True
        allowed_objective_drop = 0.0

    failures: list[str] = []
    if not finite_ok:
        failures.append("non_finite")
    if not budget_ok:
        failures.append("budget")
    if not inactive_ok:
        failures.append("inactive_leakage")
    if not qos_ok:
        failures.append("qos")
    if not objective_ok:
        failures.append("objective_drop")

    return _CandidateAcceptanceCheck(
        finite_ok=finite_ok,
        budget_ok=budget_ok,
        inactive_ok=inactive_ok,
        qos_ok=qos_ok,
        objective_ok=objective_ok,
        min_qos_margin=min_qos_margin,
        transformed_objective=transformed_objective,
        allowed_objective_drop=allowed_objective_drop,
        accepted=(len(failures) == 0),
        message=("ok" if len(failures) == 0 else ",".join(failures)),
    )



def _solve_cvxpy_problem_with_acceptance(
    prob: Any,
    rho_var: Any,
    t_var: Any,
    ctx: _SetupOptimizationContext,
    gamma_qos: np.ndarray,
    alpha: float,
    reference_transformed_objective: float,
    options: OptimizedPowerOptions,
    log_label: str = "",
) -> tuple[Optional[np.ndarray], Optional[_SetupTrueMetrics], str, str, Optional[_CandidateAcceptanceCheck]]:
    """
    Solve a CVXPY problem with fallback solvers and accept only post-checked candidates.

    Optimal points are accepted only if they pass the hard numerical checks.
    Optimal_inaccurate points are accepted only if they also pass the objective-drop guard.
    """
    solvers_to_try = [options.solver, *options.fallback_solvers]
    last_error: Exception | None = None
    last_status = "no_status"
    last_solver = "NONE"

    best_inaccurate_candidate: tuple[np.ndarray, _SetupTrueMetrics, str, str, _CandidateAcceptanceCheck] | None = None
    best_inaccurate_objective = -np.inf

    for solver_name in solvers_to_try:
        try:
            _debug_log(
                options,
                f"[SOLVER START] {log_label} solver={solver_name} verbose={options.solver_verbose}"
            )

            if options.debug_enabled and options.debug_capture_solver_output and options.debug_log_path is not None:
                log_path = Path(options.debug_log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
                    prob.solve(
                        solver=solver_name,
                        verbose=options.solver_verbose,
                        warm_start=options.warm_start,
                    )
            else:
                prob.solve(
                    solver=solver_name,
                    verbose=options.solver_verbose,
                    warm_start=options.warm_start,
                )

            last_status = str(prob.status)
            last_solver = str(solver_name)

            _debug_log(
                options,
                f"[SOLVER END] {log_label} solver={solver_name} status={prob.status} value={prob.value}"
            )

        except Exception as exc:
            last_error = exc
            _debug_log(
                options,
                f"[SOLVER EXCEPTION] {log_label} solver={solver_name} error={exc}"
            )
            continue

        status = str(prob.status)
        if status not in ("optimal", "optimal_inaccurate"):
            continue

        if rho_var.value is None or t_var.value is None:
            _debug_log(
                options,
                f"[CANDIDATE CHECK] {log_label} solver={solver_name} status={status} accepted=0 reason=missing_values"
            )
            continue

        rho_candidate = np.asarray(rho_var.value, dtype=float)
        rho_candidate = np.maximum(rho_candidate, 0.0)
        rho_candidate = rho_candidate * ctx.serve_mask.astype(float)

        metrics_candidate = _evaluate_setup_true_metrics(ctx, rho_candidate)
        postcheck = _evaluate_candidate_acceptance(
            ctx=ctx,
            metrics=metrics_candidate,
            gamma_qos=gamma_qos,
            alpha=alpha,
            reference_transformed_objective=reference_transformed_objective,
            options=options,
            require_objective_guard=(status == "optimal_inaccurate"),
        )

        _debug_log(
            options,
            "[CANDIDATE CHECK] "
            f"{log_label} solver={solver_name} status={status} accepted={int(postcheck.accepted)} "
            f"finite={int(postcheck.finite_ok)} budget={int(postcheck.budget_ok)} "
            f"inactive={int(postcheck.inactive_ok)} qos={int(postcheck.qos_ok)} "
            f"objective={int(postcheck.objective_ok)} min_qos_margin={postcheck.min_qos_margin:.3e} "
            f"candidate_obj={postcheck.transformed_objective:.6e} "
            f"reference_obj={reference_transformed_objective:.6e} "
            f"allowed_drop={postcheck.allowed_objective_drop:.6e} "
            f"reason={postcheck.message}"
        )

        if status == "optimal":
            if postcheck.accepted:
                return rho_candidate, metrics_candidate, status, str(solver_name), postcheck
            continue

        if options.accept_inaccurate_with_postcheck and postcheck.accepted:
            if postcheck.transformed_objective > best_inaccurate_objective:
                best_inaccurate_objective = postcheck.transformed_objective
                best_inaccurate_candidate = (
                    rho_candidate,
                    metrics_candidate,
                    status,
                    str(solver_name),
                    postcheck,
                )

    if best_inaccurate_candidate is not None:
        return best_inaccurate_candidate

    if last_error is not None:
        _debug_log(
            options,
            f"[SOLVER WARNING] {log_label} no acceptable candidate from any solver. last_error={last_error}"
        )
    else:
        _debug_log(
            options,
            f"[SOLVER WARNING] {log_label} no acceptable candidate from any solver. "
            f"last_status={last_status} last_solver={last_solver}"
        )

    return None, None, "no_acceptable_candidate", "NONE", None



def _classify_power_stop_reason(
    *,
    converged_dinkelbach: bool,
    stopped_on_rejected_candidate: bool,
    num_dinkelbach_iters: int,
    max_dinkelbach_iters: int,
    final_qos_feasible: bool,
) -> str:
    """
    Classify why the Dinkelbach power loop stopped.

    This is for reporting only. It does not change the optimizer.
    """
    if converged_dinkelbach:
        return "converged"

    if stopped_on_rejected_candidate:
        return "rejected_candidate"

    if int(num_dinkelbach_iters) >= int(max_dinkelbach_iters):
        return "hit_max_dinkelbach_iterations"

    if not final_qos_feasible:
        return "final_qos_infeasible"

    return "stopped_without_convergence"



def _solve_one_setup_optimized(
    ctx: _SetupOptimizationContext,
    options: OptimizedPowerOptions,
) -> tuple[np.ndarray, np.ndarray | None, OptimizedPowerDebugSetup]:
    """
    Solve the QoS-constrained EE maximization for one setup.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "CVXPY is required for scheme='optimized'. Install it with 'pip install cvxpy' "
            "and, if needed, install basic solvers with 'pip install ecos scs'."
        ) from exc

    L, K = ctx.serve_mask.shape
    safe_floor = float(options.safe_user_power_floor_watt)
    if safe_floor <= 0.0:
        raise ValueError("safe_user_power_floor_watt must be positive")

    num_served_per_ap = np.sum(ctx.serve_mask, axis=1).astype(int)
    denom = np.maximum(num_served_per_ap.astype(float), 1.0)
    eta_init = (ctx.pmax_watt_per_ap / denom)[:, None] * ctx.serve_mask.astype(float)
    rho_cur = np.sqrt(np.maximum(eta_init, 0.0))

    metrics_cur = _evaluate_setup_true_metrics(ctx, rho_cur)
    gamma_qos = _build_qos_target_vector(metrics_cur.sinr, options)

    initial_feasible = bool(np.all(metrics_cur.sinr >= gamma_qos - options.qos_tol))
    if not initial_feasible:
        raise RuntimeError(
            "Equal-power initialization is not feasible for the requested QoS target. "
            "For scaled_equal_baseline mode this should not happen when 0 <= qos_kappa <= 1."
        )

    if options.debug_enabled:
        _debug_log(
            options,
            "[QOS TARGET] "
            f"setup={ctx.setup_index} mode={options.qos_target_mode} "
            f"qos_kappa={options.qos_kappa:.3f} "
            f"gamma_qos={np.array2string(gamma_qos, precision=3, separator=', ')}"
        )

    init_user_total_power, zero_users_before = _user_total_power_report(
        metrics_cur.eta,
        options.debug_zero_power_tol,
    )
    init_user_total_safe, init_users_below_safe, init_min_user_power, init_all_above_safe = _power_floor_status(
        metrics_cur.eta,
        safe_floor,
    )

    if options.debug_enabled and options.debug_zero_power_users:
        _debug_log(
            options,
            "[ZERO POWER BEFORE] "
            f"setup={ctx.setup_index} zero_users={zero_users_before} "
            f"user_total_power={np.array2string(init_user_total_power, precision=3, separator=', ')}"
        )

    alpha = float(metrics_cur.ee_bit_per_joule)

    alpha_history: list[float] = []
    dinkelbach_residual_history: list[float] = []
    sca_rho_change_history: list[float] = []
    min_qos_margin_history: list[float] = [float(np.min(metrics_cur.sinr - gamma_qos))]
    objective_history: list[float] = [_transformed_objective_value(metrics_cur, alpha)]
    min_user_total_power_history: list[float] = [init_min_user_power]
    num_users_below_safe_floor_history: list[int] = [len(init_users_below_safe)]
    solver_status_history: list[str] = []
    solver_name_history: list[str] = []

    converged_dinkelbach = False
    converged_sca_last_outer = False
    num_sca_iters_last_outer = 0
    stopped_on_rejected_candidate = False

    safe_point_found = init_all_above_safe
    eta_safe = metrics_cur.eta.copy() if safe_point_found else None
    safe_metrics = metrics_cur if safe_point_found else None
    safe_user_total_power = init_user_total_safe.copy() if safe_point_found else None
    safe_point_outer_iter: Optional[int] = 0 if safe_point_found else None
    safe_point_sca_iter: Optional[int] = 0 if safe_point_found else None

    log2_inv = 1.0 / np.log(2.0)

    _report_log(options, "=" * 88)
    _report_log(
        options,
        "[SETUP START] "
        f"setup={ctx.setup_index} safe_floor={safe_floor:.6e} "
        f"initial_feasible={int(initial_feasible)} active_aps={int(np.sum(ctx.delta_ap_active))} "
        f"served_links={int(np.sum(ctx.serve_mask))}"
    )
    _report_log(
        options,
        "[INIT POINT] "
        f"setup={ctx.setup_index} "
        + _brief_point_summary(metrics_cur, gamma_qos, init_user_total_safe, safe_floor)
        + f" safe_point_saved={int(safe_point_found)}"
    )

    qmat_wrapped: list[list[Any | None]] = [[None for _ in range(K)] for _ in range(K)]
    for k in range(K):
        for u in range(K):
            if ctx.q_nonzero[k, u]:
                qmat_wrapped[k][u] = cp.psd_wrap(ctx.q_mats[k, u])

    for outer_idx in range(int(options.max_dinkelbach_iters)):
        alpha_history.append(alpha)
        omega = ctx.bandwidth_hz * (
            1.0 - alpha * ctx.fronthaul_energy_per_bit_joule * ctx.user_serving_count.astype(float)
        )

        converged_sca_last_outer = False

        for sca_idx in range(int(options.max_sca_iters)):
            num_sca_iters_last_outer = sca_idx + 1

            metrics_cur = _evaluate_setup_true_metrics(ctx, rho_cur)
            d_bar = np.sum(ctx.ds_coeff * rho_cur, axis=0)
            psi_bar = np.maximum(metrics_cur.psi, ctx.eps)
            t_cur = np.maximum(metrics_cur.sinr, gamma_qos)

            rho_var = cp.Variable((L, K), nonneg=True)
            t_var = cp.Variable(K)

            constraints: list[Any] = []

            constraints.append(cp.multiply((~ctx.serve_mask).astype(float), rho_var) == 0.0)

            for l in range(L):
                constraints.append(cp.sum_squares(rho_var[l, :]) <= ctx.pmax_watt_per_ap)

            for k in range(K):
                d_expr = ctx.ds_coeff[:, k] @ rho_var[:, k]

                psi_expr: Any = 1.0
                for u in range(K):
                    if ctx.q_nonzero[k, u]:
                        psi_expr = psi_expr + cp.quad_form(rho_var[:, u], qmat_wrapped[k][u])

                gamma_lb = (
                    (d_bar[k] ** 2) / psi_bar[k]
                    + (2.0 * d_bar[k] / psi_bar[k]) * (d_expr - d_bar[k])
                    - ((d_bar[k] ** 2) / (psi_bar[k] ** 2)) * (psi_expr - psi_bar[k])
                )

                constraints.append(t_var[k] >= gamma_qos[k])
                constraints.append(t_var[k] <= gamma_lb)

            objective_expr: Any = 0.0
            for k in range(K):
                if omega[k] >= 0.0:
                    objective_expr = (
                        objective_expr
                        + omega[k] * ctx.prelog_factor * log2_inv * cp.log1p(t_var[k])
                    )
                else:
                    rate_cur = ctx.prelog_factor * np.log2(1.0 + t_cur[k])
                    slope_cur = omega[k] * ctx.prelog_factor * log2_inv / (1.0 + t_cur[k])
                    objective_expr = objective_expr + omega[k] * rate_cur + slope_cur * (t_var[k] - t_cur[k])

            power_penalty = alpha * (ctx.prelog_factor / ctx.pa_efficiency) * cp.sum(
                cp.multiply(ctx.delta_ap_active, cp.sum(cp.square(rho_var), axis=1))
            )
            objective_expr = objective_expr - power_penalty

            problem = cp.Problem(cp.Maximize(objective_expr), constraints)

            is_dcp = bool(problem.is_dcp())
            _debug_log(
                options,
                f"[PROBLEM CHECK] setup={ctx.setup_index}_outer={outer_idx}_sca={sca_idx} is_dcp={int(is_dcp)}"
            )
            if not is_dcp:
                raise RuntimeError(
                    f"Constructed CVXPY problem is not DCP for setup={ctx.setup_index}, "
                    f"outer={outer_idx}, sca={sca_idx}. Check preceding QMAT lines."
                )

            reference_transformed_objective = _transformed_objective_value(metrics_cur, alpha)

            rho_new, metrics_new, solver_status, solver_name, _ = _solve_cvxpy_problem_with_acceptance(
                problem,
                rho_var,
                t_var,
                ctx=ctx,
                gamma_qos=gamma_qos,
                alpha=alpha,
                reference_transformed_objective=reference_transformed_objective,
                options=options,
                log_label=f"setup={ctx.setup_index}_outer={outer_idx}_sca={sca_idx}",
            )
            solver_status_history.append(solver_status)
            solver_name_history.append(solver_name)

            if rho_new is None or metrics_new is None:
                stopped_on_rejected_candidate = True
                _debug_log(
                    options,
                    "[SCA WARNING] "
                    f"setup={ctx.setup_index} outer={outer_idx} sca={sca_idx} "
                    "no acceptable candidate was returned. Keeping the previous accepted point."
                )
                _report_log(
                    options,
                    "[SCA ITER] "
                    f"setup={ctx.setup_index} outer={outer_idx + 1} sca={sca_idx + 1} "
                    f"accepted=0 solver={solver_name} status={solver_status} reason=no_acceptable_candidate"
                )
                break

            transformed_obj = _transformed_objective_value(metrics_new, alpha)
            rho_change = float(
                np.linalg.norm(rho_new - rho_cur) / max(np.linalg.norm(rho_cur), options.numeric_eps)
            )

            user_total_safe, users_below_safe, min_user_power, all_above_safe = _power_floor_status(
                metrics_new.eta,
                safe_floor,
            )
            min_qos_margin = float(np.min(metrics_new.sinr - gamma_qos))

            sca_rho_change_history.append(rho_change)
            min_qos_margin_history.append(min_qos_margin)
            objective_history.append(transformed_obj)
            min_user_total_power_history.append(min_user_power)
            num_users_below_safe_floor_history.append(len(users_below_safe))

            safe_point_updated = False
            if all_above_safe:
                eta_safe = metrics_new.eta.copy()
                safe_metrics = metrics_new
                safe_user_total_power = user_total_safe.copy()
                safe_point_found = True
                safe_point_outer_iter = outer_idx + 1
                safe_point_sca_iter = sca_idx + 1
                safe_point_updated = True

            _report_log(
                options,
                "[SCA ITER] "
                f"setup={ctx.setup_index} outer={outer_idx + 1} sca={sca_idx + 1} "
                f"accepted=1 solver={solver_name} status={solver_status} "
                f"rho_change={rho_change:.3e} objective={transformed_obj:.6e} "
                + _brief_point_summary(metrics_new, gamma_qos, user_total_safe, safe_floor)
                + f" safe_point_updated={int(safe_point_updated)}"
            )

            rho_cur = rho_new
            metrics_cur = metrics_new

            if rho_change <= options.sca_tol:
                converged_sca_last_outer = True
                break

        residual = _transformed_objective_value(metrics_cur, alpha)
        relative_residual = abs(residual) / max(abs(metrics_cur.sum_rate_bps), options.numeric_eps)
        dinkelbach_residual_history.append(float(relative_residual))

        _report_log(
            options,
            "[OUTER SUMMARY] "
            f"setup={ctx.setup_index} outer={outer_idx + 1} alpha={alpha:.6e} "
            f"relative_residual={relative_residual:.3e} converged_sca={int(converged_sca_last_outer)} "
            f"stopped_on_rejected_candidate={int(stopped_on_rejected_candidate)}"
        )

        if stopped_on_rejected_candidate:
            break

        if relative_residual <= options.dinkelbach_tol:
            converged_dinkelbach = True
            break

        alpha = float(metrics_cur.sum_rate_bps / max(metrics_cur.total_power_watt, options.numeric_eps))

    final_qos_feasible = True
    final_min_qos_margin = float(np.min(metrics_cur.sinr - gamma_qos))
    final_warning_message = None

    if options.validate_final_result:
        final_qos_feasible, final_min_qos_margin = _validate_optimized_result_one_setup(
            ctx=ctx,
            metrics=metrics_cur,
            gamma_qos=gamma_qos,
            options=options,
        )

        if not final_qos_feasible:
            final_warning_message = (
                f"Setup {ctx.setup_index} finished with a QoS-infeasible final point. "
                f"min_qos_margin={final_min_qos_margin:.3e}"
            )
            _debug_log(
                options,
                "[FINAL QOS WARNING] "
                f"setup={ctx.setup_index} min_qos_margin={final_min_qos_margin:.3e}"
            )
            if options.warn_on_final_qos_infeasible:
                warnings.warn(final_warning_message, RuntimeWarning, stacklevel=2)

    final_user_total_power, zero_users_after = _user_total_power_report(
        metrics_cur.eta,
        options.debug_zero_power_tol,
    )
    final_user_total_safe, final_users_below_safe, final_min_user_power, _ = _power_floor_status(
        metrics_cur.eta,
        safe_floor,
    )

    if options.debug_enabled and options.debug_zero_power_users:
        _debug_log(
            options,
            "[ZERO POWER AFTER] "
            f"setup={ctx.setup_index} zero_users={zero_users_after} "
            f"user_total_power={np.array2string(final_user_total_power, precision=3, separator=', ')}"
        )

    if safe_metrics is not None and safe_user_total_power is not None:
        safe_point_summary = _brief_point_summary(safe_metrics, gamma_qos, safe_user_total_power, safe_floor)
        delta_ee = metrics_cur.ee_bit_per_joule - safe_metrics.ee_bit_per_joule
        delta_rate = metrics_cur.sum_rate_bps - safe_metrics.sum_rate_bps
        delta_power = metrics_cur.total_power_watt - safe_metrics.total_power_watt
    else:
        safe_point_summary = "not_found"
        delta_ee = np.nan
        delta_rate = np.nan
        delta_power = np.nan

    stop_reason = _classify_power_stop_reason(
        converged_dinkelbach=bool(converged_dinkelbach),
        stopped_on_rejected_candidate=bool(stopped_on_rejected_candidate),
        num_dinkelbach_iters=len(alpha_history),
        max_dinkelbach_iters=int(options.max_dinkelbach_iters),
        final_qos_feasible=bool(final_qos_feasible),
    )

    _report_log(
        options,
        "[FINAL POINT] "
        f"setup={ctx.setup_index} converged_dinkelbach={int(converged_dinkelbach)} "
        f"converged_sca_last_outer={int(converged_sca_last_outer)} "
        f"num_outer_iters={len(alpha_history)} num_sca_iters_last_outer={num_sca_iters_last_outer} "
        + _brief_point_summary(metrics_cur, gamma_qos, final_user_total_safe, safe_floor)
    )
    if safe_metrics is not None and safe_user_total_power is not None:
        _report_log(
            options,
            "[SAFE POINT] "
            f"setup={ctx.setup_index} found=1 outer={safe_point_outer_iter} sca={safe_point_sca_iter} "
            + safe_point_summary
        )
        _report_log(
            options,
            "[FINAL VS SAFE] "
            f"setup={ctx.setup_index} delta_ee={delta_ee:.6e} "
            f"delta_sum_rate={delta_rate:.6e} delta_total_power={delta_power:.6e}"
        )
    else:
        _report_log(
            options,
            f"[SAFE POINT] setup={ctx.setup_index} found=0 safe_floor={safe_floor:.6e}"
        )
    _report_log(options, "=" * 88)

    debug = OptimizedPowerDebugSetup(
        initial_feasible=initial_feasible,
        used_equal_power_init=True,
        alpha_history=alpha_history,
        dinkelbach_residual_history=dinkelbach_residual_history,
        sca_rho_change_history=sca_rho_change_history,
        min_qos_margin_history=min_qos_margin_history,
        objective_history=objective_history,
        min_user_total_power_history=min_user_total_power_history,
        num_users_below_safe_floor_history=num_users_below_safe_floor_history,
        solver_status_history=solver_status_history,
        solver_name_history=solver_name_history,
        converged_dinkelbach=converged_dinkelbach,
        converged_sca_last_outer=converged_sca_last_outer,
        num_dinkelbach_iters=len(alpha_history),
        num_sca_iters_last_outer=num_sca_iters_last_outer,
        init_user_total_power=init_user_total_power.tolist(),
        final_user_total_power=final_user_total_power.tolist(),
        zero_power_users_before=zero_users_before,
        zero_power_users_after=zero_users_after,
        qos_target=gamma_qos.tolist(),
        safe_user_power_floor_watt=safe_floor,
        safe_point_found=safe_point_found,
        safe_point_outer_iter=safe_point_outer_iter,
        safe_point_sca_iter=safe_point_sca_iter,
        safe_point_user_total_power=(safe_user_total_power.tolist() if safe_user_total_power is not None else None),
        safe_point_ee_bit_per_joule=(float(safe_metrics.ee_bit_per_joule) if safe_metrics is not None else None),
        safe_point_sum_rate_bps=(float(safe_metrics.sum_rate_bps) if safe_metrics is not None else None),
        safe_point_total_power_watt=(float(safe_metrics.total_power_watt) if safe_metrics is not None else None),
        safe_point_min_qos_margin=(float(np.min(safe_metrics.sinr - gamma_qos)) if safe_metrics is not None else None),
        safe_point_transformed_objective=(
            float(_transformed_objective_value(safe_metrics, alpha_history[min(max((safe_point_outer_iter or 1) - 1, 0), len(alpha_history) - 1)]))
            if safe_metrics is not None and len(alpha_history) > 0 else None
        ),
        safe_point_min_user_total_power=(float(np.min(safe_user_total_power)) if safe_user_total_power is not None else None),
        safe_point_num_users_below_safe_floor=(0 if safe_user_total_power is not None else None),
        final_qos_feasible=final_qos_feasible,
        final_min_qos_margin=final_min_qos_margin,
        final_min_user_total_power=final_min_user_power,
        final_num_users_below_safe_floor=len(final_users_below_safe),
        final_warning_message=final_warning_message,

        stopped_on_rejected_candidate=bool(stopped_on_rejected_candidate),
        stop_reason=stop_reason,
    )

    return metrics_cur.eta, eta_safe, debug



def _allocate_downlink_power_optimized(
    cfg: Config,
    m: np.ndarray,
    *,
    est: Any,
    corr: Any,
    ls: Any,
    pilot_of_user: np.ndarray,
    phases: Any,
    pilot_power_watt_per_user: Optional[np.ndarray],
    options: OptimizedPowerOptions,
) -> PowerAllocationSetups:
    """
    Optimized power allocation over all setups.
    """
    S, L, K = m.shape

    _reset_report_log(options)
    _report_log(
        options,
        "[RUN START] "
        f"num_setups={S} safe_floor={float(options.safe_user_power_floor_watt):.6e}"
    )

    common = _build_common_closed_form_data(
        cfg=cfg,
        est=est,
        corr=corr,
        ls=ls,
        pilot_of_user=pilot_of_user,
        phases=phases,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    eta_all = np.zeros((S, L, K), dtype=float)
    eta_safe_all = np.full((S, L, K), np.nan, dtype=float)
    safe_point_found_any = False
    debug_all: list[OptimizedPowerDebugSetup] = []

    for s in range(S):
        ctx = _build_setup_optimization_context(
            cfg=cfg,
            setup_index=s,
            serve_mask_s=m[s],
            pilot_of_user_s=np.asarray(pilot_of_user[s], dtype=int),
            common=common,
            options=options,
        )
        eta_s, eta_safe_s, debug_s = _solve_one_setup_optimized(ctx, options)
        eta_all[s] = eta_s
        if eta_safe_s is not None:
            eta_safe_all[s] = eta_safe_s
            safe_point_found_any = True
        debug_all.append(debug_s)

    ap_total_power = np.sum(eta_all, axis=2)
    eta_sqrt = np.where(eta_all > 0.0, np.sqrt(eta_all), 0.0)
    num_served = np.sum(m, axis=2).astype(int)

    Pmax = float(cfg.power.dl_max_power_watt_per_ap)
    if np.any(ap_total_power > Pmax + 1e-7):
        raise RuntimeError("Optimized power allocation violates the per-AP power budget.")

    safe_count = sum(int(d.safe_point_found) for d in debug_all)
    _report_log(
        options,
        f"[RUN END] num_setups={S} safe_points_found={safe_count}/{S}"
    )

    return PowerAllocationSetups(
        eta=eta_all,
        eta_sqrt=eta_sqrt,
        num_served_users=num_served,
        ap_total_power=ap_total_power,
        serve_mask_used=m,
        scheme="optimized",
        pmax_watt_per_ap=Pmax,
        optimized_debug_by_setup=debug_all,
        eta_safe=(eta_safe_all if safe_point_found_any else None),
    )



def allocate_downlink_power(
    cfg: Config,
    scheme: PowerScheme = "equal",
    *,
    serve_mask: Optional[np.ndarray] = None,
    serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    est: Any | None = None,
    corr: Any | None = None,
    ls: Any | None = None,
    pilot_of_user: Optional[np.ndarray] = None,
    phases: Any | None = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    optimized_options: Optional[OptimizedPowerOptions] = None,
) -> PowerAllocationSetups:
    """
    Allocate downlink power coefficients eta.
    """
    cfg.validate()

    Pmax = float(cfg.power.dl_max_power_watt_per_ap)
    if Pmax < 0:
        raise ValueError("dl_max_power_watt_per_ap must be non-negative")

    m = _validate_or_build_serve_mask(cfg, serve_mask=serve_mask, serve_users=serve_users)

    if scheme == "equal":
        return _allocate_downlink_power_equal(cfg, m)

    if scheme == "optimized":
        if est is None or corr is None or ls is None or pilot_of_user is None:
            raise ValueError(
                "scheme='optimized' requires est, corr, ls, and pilot_of_user. "
                "These are needed to extract the closed-form SINR coefficients."
            )

        options = OptimizedPowerOptions() if optimized_options is None else optimized_options
        return _allocate_downlink_power_optimized(
            cfg,
            m,
            est=est,
            corr=corr,
            ls=ls,
            pilot_of_user=pilot_of_user,
            phases=phases,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
            options=options,
        )

    raise ValueError(f"Unknown scheme {scheme}")