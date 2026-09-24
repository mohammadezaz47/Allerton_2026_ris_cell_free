# ITNG YMYFA 80asra YA
# IMZZ

# src/association_game_optimizer.py

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import itertools
import numpy as np

from .config import Config
from .geometry import GeometrySetups
from .large_scale import LargeScaleSetups
from .correlation_setups import CorrelationSetups
from .ris_phase_shifts import PhaseShiftSetups
from .pilot_assignment import PilotAssignmentSetups
from .power_allocation import PowerAllocationSetups, allocate_downlink_power
from .channel_estimation import (
    ChannelEstimationSetups,
    build_channel_estimation_statistics,
)
from .dl_sinr_closed_form import (
    DownlinkSinrClosedFormSetups,
    compute_dl_sinr_closed_form,
)
from .spectral_efficiency import (
    SpectralEfficiencySetups,
    compute_spectral_efficiency_setups,
)
from .energy_efficiency import (
    EnergyEfficiencySetups,
    compute_energy_efficiency_setups,
)


RedistributionScheme = Literal["equal"]
InitActionMode = Literal["full_candidate", "random", "strongest_singleton"]
ResponseMode = Literal["best_response", "logit"]


@dataclass(frozen=True)
class GameOptions:
    """
    Options that control the association game optimizer.

    Design policy
    - The optimizer keeps the outer Dinkelbach iterations.
    - Inside the game, every serving mask is evaluated using equal-power
      redistribution induced by that mask.
    - The default update is deterministic best response. Logit is kept only
      as an optional exploration mode.
    - No optimized-power call is allowed inside this file.
    """
    d_th: float

    redistribution_scheme: RedistributionScheme = "equal"
    init_action_mode: InitActionMode = "strongest_singleton"
    response_mode: ResponseMode = "best_response"

    omega_initial: float = 4.0
    max_game_iters: int = 50
    max_dinkelbach_iters: int = 10

    dinkelbach_tol: float = 1e-3
    convergence_window: int = 3
    stability_tol: float = 1e-3
    utility_improvement_tol: float = 1e-12

    max_subset_size: int | None = None
    max_actions_per_user: int | None = None

    incumbent_tracking: bool = True
    outer_warm_start: bool = True
    cycle_detection: bool = True

    random_seed: int | None = None
    alpha_init: float = 0.0
    enforce_user_served: bool = True

    def validate(self) -> None:
        if self.d_th <= 0:
            raise ValueError("d_th must be positive")
        if self.redistribution_scheme != "equal":
            raise ValueError("Only redistribution_scheme='equal' is supported in this optimizer")
        if self.init_action_mode not in ("full_candidate", "random", "strongest_singleton"):
            raise ValueError("Unknown init_action_mode")
        if self.response_mode not in ("best_response", "logit"):
            raise ValueError("response_mode must be 'best_response' or 'logit'")
        if self.max_game_iters <= 0:
            raise ValueError("max_game_iters must be positive")
        if self.max_dinkelbach_iters <= 0:
            raise ValueError("max_dinkelbach_iters must be positive")
        if self.dinkelbach_tol < 0:
            raise ValueError("dinkelbach_tol must be non-negative")
        if self.convergence_window <= 0:
            raise ValueError("convergence_window must be positive")
        if self.stability_tol < 0:
            raise ValueError("stability_tol must be non-negative")
        if self.utility_improvement_tol < 0:
            raise ValueError("utility_improvement_tol must be non-negative")
        if self.omega_initial <= 0:
            raise ValueError("omega_initial must be positive")
        if self.max_subset_size is not None and self.max_subset_size <= 0:
            raise ValueError("max_subset_size must be positive when provided")
        if self.max_actions_per_user is not None and self.max_actions_per_user <= 0:
            raise ValueError("max_actions_per_user must be positive when provided")
        if not np.isfinite(self.alpha_init):
            raise ValueError("alpha_init must be finite")


@dataclass(frozen=True)
class StaticGameGraphSetup:
    """
    Static graph objects for one setup.
    """
    candidate_aps_per_user: list[np.ndarray]
    action_masks_per_user: list[list[np.ndarray]]
    neighbor_sets: list[np.ndarray]
    q_sets: list[np.ndarray]
    non_neighbor_sets: list[np.ndarray]
    maximum_non_neighbor_sets: list[np.ndarray]
    user_distance_matrix: np.ndarray
    candidate_overlap_matrix: np.ndarray


@dataclass(frozen=True)
class FixedAlphaGameSetupResult:
    """
    Result of solving the association game for one setup at a fixed alpha.
    """
    initial_serve_mask: np.ndarray              # (1, L, K)

    # This is the mask returned by the fixed-alpha game.
    # With incumbent_tracking=True, this is the best-utility mask.
    final_serve_mask: np.ndarray                # (1, L, K)

    # The actual last mask reached before returning the incumbent.
    # This is useful for debugging whether incumbent tracking changed the output.
    terminal_serve_mask: np.ndarray             # (1, L, K)

    # Best mask according to the local utility proxy.
    best_utility_serve_mask: np.ndarray         # (1, L, K)

    # Best mask according to global EE, used only as a diagnostic.
    # The game does not optimize this directly.
    best_global_ee_serve_mask: np.ndarray       # (1, L, K)

    final_power: PowerAllocationSetups
    final_sinr_cf: DownlinkSinrClosedFormSetups
    final_se: SpectralEfficiencySetups
    final_ee: EnergyEfficiencySetups

    transformed_objective_history: list[float]
    utility_proxy_history: list[float]
    ee_bit_per_joule_history: list[float]
    sum_rate_bps_history: list[float]
    total_power_watt_history: list[float]
    chosen_set_history: list[int]
    num_users_changed_history: list[int]

    best_utility_proxy: float
    best_global_ee_bit_per_joule: float

    converged: bool
    num_iterations: int
    cycle_detected: bool = False


@dataclass(frozen=True)
class AssociationOptimizerResults:
    """
    Full optimizer output over all setups.
    """
    final_serve_mask: np.ndarray                # (S, L, K)
    final_power: PowerAllocationSetups
    final_sinr_cf: DownlinkSinrClosedFormSetups
    final_se: SpectralEfficiencySetups
    final_ee: EnergyEfficiencySetups

    graphs: list[StaticGameGraphSetup]
    fixed_alpha_results_by_outer_iter: list[list[FixedAlphaGameSetupResult]]

    alpha_history: list[float]
    residual_history: list[float]
    outer_mean_ee_history: list[float]
    outer_mean_se_history: list[float]
    outer_avg_serving_aps_per_user_history: list[float]
    outer_num_users_with_mask_change_history: list[int]

    converged: bool
    num_outer_iterations: int
    options: GameOptions

# ---------------------------------------------------------------------
# Basic slicing helpers
# ---------------------------------------------------------------------

def _single_setup_cfg(cfg: Config) -> Config:
    """
    Create a copy of cfg with num_setups forced to 1.
    """
    cfg1 = replace(cfg, sim=replace(cfg.sim, num_setups=1))
    cfg1.validate()
    return cfg1


def _slice_geometry_positions(geom: GeometrySetups, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return AP, user, RIS positions for one setup.
    """
    return (
        np.asarray(geom.ap_xy[s], dtype=float),
        np.asarray(geom.user_xy[s], dtype=float),
        np.asarray(geom.ris_xy[s], dtype=float),
    )


def _slice_large_scale(ls: LargeScaleSetups, s: int) -> LargeScaleSetups:
    return LargeScaleSetups(
        ap_user_gain_db=ls.ap_user_gain_db[s:s+1],
        ap_ris_gain_db=ls.ap_ris_gain_db[s:s+1],
        ris_user_gain_db=ls.ris_user_gain_db[s:s+1],
        ap_user_beta=ls.ap_user_beta[s:s+1],
        ap_ris_beta=ls.ap_ris_beta[s:s+1],
        ris_user_beta=ls.ris_user_beta[s:s+1],
        ap_user_beta_over_noise=ls.ap_user_beta_over_noise[s:s+1],
        ap_ris_beta_over_noise=ls.ap_ris_beta_over_noise[s:s+1],
        ris_user_beta_over_noise=ls.ris_user_beta_over_noise[s:s+1],
        is_los_ap_user=ls.is_los_ap_user[s:s+1],
        is_los_ap_ris=ls.is_los_ap_ris[s:s+1],
        is_los_ris_user=ls.is_los_ris_user[s:s+1],
        is_blocked_ap_user=ls.is_blocked_ap_user[s:s+1],
        noise_power_dbm=ls.noise_power_dbm,
        noise_power_watt=ls.noise_power_watt,
    )


def _slice_corr(corr: CorrelationSetups, s: int) -> CorrelationSetups:
    return CorrelationSetups(
        delta_ap_user=corr.delta_ap_user[s:s+1],
        delta_ap_ris=corr.delta_ap_ris[s:s+1],
        R_ris=corr.R_ris,
        R_ap_user=corr.R_ap_user[s:s+1],
        R_ris_user=corr.R_ris_user[s:s+1],
        Gamma_ap_ris=corr.Gamma_ap_ris[s:s+1],
        theta_ap_user=corr.theta_ap_user[s:s+1],
        theta_ap_ris=corr.theta_ap_ris[s:s+1],
        delta_master=corr.delta_master,
    )


def _slice_phases(phases: PhaseShiftSetups, s: int) -> PhaseShiftSetups:
    return PhaseShiftSetups(
        Theta=phases.Theta[s:s+1],
        vartheta=phases.vartheta[s:s+1],
        mode=phases.mode,
    )


def _slice_est_stats(est: ChannelEstimationSetups, s: int) -> ChannelEstimationSetups:
    return ChannelEstimationSetups(
        u_hat=None,
        R_u=est.R_u[s:s+1],
        Omega=est.Omega[s:s+1],
        Omega_inv=est.Omega_inv[s:s+1],
        W=est.W[s:s+1],
        R_hat=est.R_hat[s:s+1],
        C=est.C[s:s+1],
        pilot_of_user=est.pilot_of_user[s:s+1],
        chi=est.chi[s:s+1],
        tau_p=est.tau_p,
        pilot_power_watt_per_user=est.pilot_power_watt_per_user,
    )


def _slice_pilot_of_user(pilot_of_user: np.ndarray, s: int) -> np.ndarray:
    return np.asarray(pilot_of_user[s:s+1], dtype=int)


# ---------------------------------------------------------------------
# Static graph construction
# ---------------------------------------------------------------------

def _pairwise_user_distances(user_xy: np.ndarray) -> np.ndarray:
    """
    Pairwise 2D Euclidean distances between users in one setup.

    user_xy shape (K, 2)
    return shape (K, K)
    """
    delta = user_xy[:, None, :] - user_xy[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def _build_candidate_aps_from_mask(candidate_mask_setup: np.ndarray) -> list[np.ndarray]:
    """
    candidate_mask_setup shape (L, K)
    returns list of length K with AP-index arrays
    """
    L, K = candidate_mask_setup.shape
    out: list[np.ndarray] = []
    for k in range(K):
        aps = np.where(candidate_mask_setup[:, k])[0].astype(int)
        out.append(aps)
    return out


def _build_candidate_overlap_matrix(candidate_aps_per_user: list[np.ndarray]) -> np.ndarray:
    """
    overlap[k,k'] = True if candidate AP sets overlap
    """
    K = len(candidate_aps_per_user)
    overlap = np.zeros((K, K), dtype=bool)

    for k in range(K):
        set_k = set(candidate_aps_per_user[k].tolist())
        for kp in range(K):
            if k == kp:
                continue
            overlap[k, kp] = len(set_k.intersection(candidate_aps_per_user[kp].tolist())) > 0

    return overlap


def _build_static_neighbor_sets(
    candidate_aps_per_user: list[np.ndarray],
    user_xy: np.ndarray,
    d_th: float,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Build static neighbor sets using candidate-set overlap and distance threshold.

    N_k = {k' != k | L_k^c ∩ L_k'^c != empty and ||p_k - p_k'|| <= d_th}
    Q_k = {k} ∪ N_k
    """
    K = len(candidate_aps_per_user)
    D = _pairwise_user_distances(user_xy)                 # (K, K)
    overlap = _build_candidate_overlap_matrix(candidate_aps_per_user)  # (K, K)

    neighbors: list[np.ndarray] = []
    q_sets: list[np.ndarray] = []

    for k in range(K):
        mask = np.ones(K, dtype=bool)
        mask[k] = False
        mask &= overlap[k]
        mask &= (D[k] <= d_th)

        nk = np.where(mask)[0].astype(int)
        qk = np.unique(np.concatenate([np.array([k], dtype=int), nk])).astype(int)

        neighbors.append(nk)
        q_sets.append(qk)

    return neighbors, q_sets, D, overlap


def _build_non_neighbor_sets(neighbor_sets: list[np.ndarray], K: int) -> list[np.ndarray]:
    """
    For each user k, build the non-neighbor set with self included.

    This follows the spirit of the reference algorithm.
    """
    neighbor_mask = np.zeros((K, K), dtype=bool)
    for k in range(K):
        neighbor_mask[k, neighbor_sets[k]] = True
        neighbor_mask[k, k] = False

    non_neighbors: list[np.ndarray] = []
    for k in range(K):
        mask = ~neighbor_mask[k]
        mask[k] = True
        non_neighbors.append(np.where(mask)[0].astype(int))

    return non_neighbors


def _is_non_neighbor_with_set(user: int, current_set: list[int], neighbor_sets: list[np.ndarray]) -> bool:
    """
    Check whether 'user' is a non-neighbor of all users already in current_set.
    """
    neigh_user = set(neighbor_sets[user].tolist())
    for v in current_set:
        if v in neigh_user:
            return False
        neigh_v = set(neighbor_sets[v].tolist())
        if user in neigh_v:
            return False
    return True


def _build_maximum_non_neighbor_sets_greedy(neighbor_sets: list[np.ndarray]) -> list[np.ndarray]:
    """
    Build a family of maximal non-neighbor sets greedily.

    This mirrors the reference idea of precomputing such sets in advance.
    """
    K = len(neighbor_sets)
    unique_sets: set[tuple[int, ...]] = set()

    for seed in range(K):
        current = [seed]
        for cand in range(K):
            if cand == seed:
                continue
            if _is_non_neighbor_with_set(cand, current, neighbor_sets):
                current.append(cand)

        tup = tuple(sorted(current))
        unique_sets.add(tup)

    # Ensure every user appears in at least one set
    covered = set()
    for s in unique_sets:
        covered.update(s)

    for k in range(K):
        if k not in covered:
            unique_sets.add((k,))

    out = [np.array(s, dtype=int) for s in sorted(unique_sets, key=lambda x: (len(x), x))]
    return out


def _all_nonempty_subsets(
    elements: np.ndarray,
    max_subset_size: Optional[int] = None,
) -> list[tuple[int, ...]]:
    """
    Return all nonempty subsets of the given 1D integer array.
    """
    elems = [int(x) for x in elements.tolist()]
    n = len(elems)

    subsets: list[tuple[int, ...]] = []
    upper = n if max_subset_size is None else min(n, int(max_subset_size))

    for r in range(1, upper + 1):
        subsets.extend(itertools.combinations(elems, r))

    return subsets


def _subset_proxy_score(subset: tuple[int, ...], link_strengths: np.ndarray) -> float:
    """
    Simple proxy score used only for action pruning.
    """
    if len(subset) == 0:
        return -np.inf
    return float(np.sum(link_strengths[np.array(subset, dtype=int)]))


def _build_action_masks_for_user(
    L: int,
    candidate_aps: np.ndarray,
    link_strengths: np.ndarray,
    max_subset_size: Optional[int],
    max_actions_per_user: Optional[int],
) -> list[np.ndarray]:
    """
    Build the action space for one user as a list of boolean masks over APs.

    All actions are nonempty subsets of the candidate set.
    Optional pruning by subset size and count is supported for tractability.
    """
    subsets = _all_nonempty_subsets(candidate_aps, max_subset_size=max_subset_size)

    if max_actions_per_user is not None and len(subsets) > int(max_actions_per_user):
        ranked = sorted(
            subsets,
            key=lambda ss: (_subset_proxy_score(ss, link_strengths), len(ss)),
            reverse=True,
        )

        # Keep the best few plus some important anchors
        keep: list[tuple[int, ...]] = []

        # Best action
        keep.append(ranked[0])

        # Full candidate set if allowed by subset-size cap
        full_set = tuple(candidate_aps.tolist())
        if full_set in ranked:
            keep.append(full_set)

        # Best singleton
        best_singleton = (int(candidate_aps[np.argmax(link_strengths[candidate_aps])]),)
        if best_singleton in ranked:
            keep.append(best_singleton)

        # Fill remaining slots
        for ss in ranked:
            if ss not in keep:
                keep.append(ss)
            if len(keep) >= int(max_actions_per_user):
                break

        subsets = keep

    action_masks: list[np.ndarray] = []
    for ss in subsets:
        m = np.zeros(L, dtype=bool)
        m[np.array(ss, dtype=int)] = True
        action_masks.append(m)

    return action_masks


def build_static_game_graphs(
    cfg: Config,
    geom: GeometrySetups,
    pilot: PilotAssignmentSetups,
    *,
    d_th: float,
    max_subset_size: Optional[int] = None,
    max_actions_per_user: Optional[int] = None,
) -> list[StaticGameGraphSetup]:
    """
    Build all static graph objects and action spaces needed by the association game.
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    if d_th <= 0:
        raise ValueError("d_th must be positive")

    graphs: list[StaticGameGraphSetup] = []

    for s in range(S):
        _, user_xy, _ = _slice_geometry_positions(geom, s)
        candidate_mask_s = np.asarray(pilot.candidate_mask[s], dtype=bool)   # (L, K)

        candidate_aps_per_user = _build_candidate_aps_from_mask(candidate_mask_s)
        neighbors, q_sets, D, overlap = _build_static_neighbor_sets(candidate_aps_per_user, user_xy, d_th=d_th)
        non_neighbors = _build_non_neighbor_sets(neighbors, K)
        max_non_neighbor_sets = _build_maximum_non_neighbor_sets_greedy(neighbors)

        actions_per_user: list[list[np.ndarray]] = []
        for k in range(K):
            cand = candidate_aps_per_user[k]

            # If for any reason candidate set is empty, use master AP as fallback
            if cand.size == 0:
                cand = np.array([int(pilot.master_ap[s, k])], dtype=int)

            link_strengths = np.asarray(pilot.tr_Ru[s, :, k], dtype=float)

            actions_k = _build_action_masks_for_user(
                L=L,
                candidate_aps=cand,
                link_strengths=link_strengths,
                max_subset_size=max_subset_size,
                max_actions_per_user=max_actions_per_user,
            )
            actions_per_user.append(actions_k)

        graphs.append(
            StaticGameGraphSetup(
                candidate_aps_per_user=candidate_aps_per_user,
                action_masks_per_user=actions_per_user,
                neighbor_sets=neighbors,
                q_sets=q_sets,
                non_neighbor_sets=non_neighbors,
                maximum_non_neighbor_sets=max_non_neighbor_sets,
                user_distance_matrix=D,
                candidate_overlap_matrix=overlap,
            )
        )

    return graphs


# ---------------------------------------------------------------------
# Serve-mask and utility helpers
# ---------------------------------------------------------------------

def _actions_to_serve_mask(action_masks_per_user: list[np.ndarray]) -> np.ndarray:
    """
    Convert a list of K action masks of shape (L,) into a serving mask of shape (1, L, K).
    """
    K = len(action_masks_per_user)
    L = int(action_masks_per_user[0].shape[0])

    out = np.zeros((1, L, K), dtype=bool)
    for k in range(K):
        out[0, :, k] = action_masks_per_user[k]
    return out


def _all_users_served(serve_mask: np.ndarray) -> bool:
    """
    Return True iff every user is served by at least one AP.
    """
    served_counts = np.sum(np.asarray(serve_mask, dtype=bool), axis=1)
    return bool(np.all(served_counts >= 1))


def _validate_all_users_served(serve_mask: np.ndarray, *, where: str) -> None:
    """
    Raise if any user is left unserved.
    """
    if _all_users_served(serve_mask):
        return

    served_counts = np.sum(np.asarray(serve_mask, dtype=bool), axis=1)
    missing = np.where(served_counts[0] <= 0)[0].astype(int).tolist()
    raise RuntimeError(f"Users left unserved at {where}. Missing user indices: {missing}")


def _evaluate_profile_global(
    cfg: Config,
    est_stats: ChannelEstimationSetups,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    phases: PhaseShiftSetups,
    pilot_of_user: np.ndarray,
    serve_mask: np.ndarray,
    redistribution_scheme: RedistributionScheme,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
) -> tuple[PowerAllocationSetups, DownlinkSinrClosedFormSetups, SpectralEfficiencySetups, EnergyEfficiencySetups, float]:
    """
    Evaluate one full serving profile globally using the closed-form branch.

    Design policy
    - Every serving mask is evaluated using the equal-power redistribution
      induced by that mask.
    - No optimized-power call is allowed inside this file.
    """
    if redistribution_scheme != "equal":
        raise ValueError(f"Unknown redistribution_scheme {redistribution_scheme}")

    power_out = allocate_downlink_power(
        cfg,
        scheme="equal",
        serve_mask=serve_mask,
    )

    sinr_cf = compute_dl_sinr_closed_form(
        cfg=cfg,
        est=est_stats,
        corr=corr,
        ls=ls,
        power=power_out,
        pilot_of_user=pilot_of_user,
        phases=phases,
        serve_mask=serve_mask,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    se_out = compute_spectral_efficiency_setups(cfg, sinr_cf.sinr_dl)
    ee_out = compute_energy_efficiency_setups(cfg, se_out, power_out, serve_mask=serve_mask)

    transformed_obj_without_alpha = float(np.sum(se_out.rate_bps))
    return power_out, sinr_cf, se_out, ee_out, transformed_obj_without_alpha


def _evaluate_local_utility(
    cfg: Config,
    q_set_k: np.ndarray,
    alpha: float,
    serve_mask: np.ndarray,
    power_out: PowerAllocationSetups,
    se_out: SpectralEfficiencySetups,
    ee_out: EnergyEfficiencySetups,
) -> float:
    """
    Evaluate the local altruistic utility for one user k using already computed global outputs.

      U_k^alpha
      = B sum_{i in Q_k} SE_i
        - alpha [ sum_{l in L_k^loc} (P_tx,l + P_c delta_l)
                  + B E_p sum_{i in Q_k} |L_i| SE_i ]

    The RIS static term is omitted since it is constant across action profiles.
    """
    q_idx = np.asarray(q_set_k, dtype=int).ravel()

    # Local AP set = union of serving APs of users in Q_k
    local_ap_mask = np.any(serve_mask[0, :, q_idx], axis=1)  # (L,)
    local_ap_idx = np.where(local_ap_mask)[0].astype(int)

    B = float(cfg.noise.bandwidth_hz)
    P_c = float(cfg.ee.ap_circuit_power_watt)
    E_p = float(cfg.ee.fronthaul_energy_per_bit_joule)

    se_local = np.asarray(se_out.se[0, q_idx], dtype=float)
    q_serving_counts = np.asarray(ee_out.user_serving_count[0, q_idx], dtype=float)

    tx_local = float(np.sum(ee_out.ap_tx_power_watt[0, local_ap_idx]))
    circuit_local = float(P_c * np.sum(ee_out.ap_active_mask[0, local_ap_idx].astype(float)))
    fronthaul_local = float(B * E_p * np.sum(q_serving_counts * se_local))

    utility = float(np.sum(se_out.rate_bps[0, q_idx]) - alpha * (tx_local + circuit_local + fronthaul_local))
    return utility


def _global_transformed_objective(alpha: float, ee_out: EnergyEfficiencySetups) -> float:
    """
    Global Dinkelbach-transformed objective used only as a diagnostic.
      B sum_k SE_k - alpha P_T
    """
    return float(ee_out.sum_rate_bps[0] - alpha * ee_out.total_power_watt[0])


def _relative_change(current: float, previous: float, eps: float = 1e-12) -> float:
    """
    Symmetric relative change.
    """
    current = float(current)
    previous = float(previous)
    if not np.isfinite(current) or not np.isfinite(previous):
        return float("inf")
    denom = max(abs(current), abs(previous), float(eps))
    return float(abs(current - previous) / denom)


def _profile_utility_proxy(
    cfg: Config,
    graph: StaticGameGraphSetup,
    alpha: float,
    serve_mask: np.ndarray,
    power_out: PowerAllocationSetups,
    se_out: SpectralEfficiencySetups,
    ee_out: EnergyEfficiencySetups,
) -> float:
    """
    Utility-based profile score used for convergence and incumbent tracking.

    This is not the global EE. It is the sum of the local altruistic utilities
    evaluated under the current profile. It keeps the implementation aligned
    with the local game interpretation.
    """
    total = 0.0
    for q_set in graph.q_sets:
        total += _evaluate_local_utility(
            cfg=cfg,
            q_set_k=q_set,
            alpha=alpha,
            serve_mask=serve_mask,
            power_out=power_out,
            se_out=se_out,
            ee_out=ee_out,
        )
    return float(total)


def _mask_key(mask: np.ndarray) -> bytes:
    """
    Compact key for cycle detection.
    """
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _find_action_index(actions_k: list[np.ndarray], action_mask: np.ndarray) -> int | None:
    for j, a in enumerate(actions_k):
        if np.array_equal(a, action_mask):
            return int(j)
    return None


def _actions_from_serve_mask(
    graph: StaticGameGraphSetup,
    serve_mask: np.ndarray,
) -> list[np.ndarray]:
    """
    Convert a serving mask into per-user action masks.

    The mask should have shape (1, L, K) or (L, K). The selected action for
    each user must be present in the user's action space.
    """
    m = np.asarray(serve_mask, dtype=bool)
    if m.ndim == 3:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"warm-start mask must have shape (1,L,K) or (L,K), got {m.shape}")

    K = len(graph.action_masks_per_user)
    actions: list[np.ndarray] = []
    for k in range(K):
        action_k = m[:, k]
        if _find_action_index(graph.action_masks_per_user[k], action_k) is None:
            raise ValueError(f"warm-start action for user {k} is not in the action space")
        actions.append(action_k.copy())
    return actions


# ---------------------------------------------------------------------
# Single-setup fixed-alpha game
# ---------------------------------------------------------------------

def _omega_schedule(iter_idx: int, omega_initial: float) -> float:
    """
    Logit exploration schedule inspired by the reference algorithm.
    """
    if iter_idx <= 1:
        return float(omega_initial)
    return float(omega_initial) * np.log(1.0 + float(iter_idx))


def _sample_logit_action(
    utilities: np.ndarray,
    omega: float,
    rng: np.random.Generator,
) -> int:
    """
    Sample an action index according to the logit rule.
    """
    u = np.asarray(utilities, dtype=float)
    logits = omega * u
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(rng.choice(np.arange(u.shape[0]), p=probs))


def _initialize_actions_for_setup(
    cfg: Config,
    graph: StaticGameGraphSetup,
    pilot: PilotAssignmentSetups,
    s: int,
    init_action_mode: InitActionMode,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Choose an initial action for each user in one setup.
    """
    K = int(cfg.dims.num_users)
    action_masks: list[np.ndarray] = []

    for k in range(K):
        actions_k = graph.action_masks_per_user[k]
        if len(actions_k) == 0:
            raise RuntimeError("Empty action set encountered")

        if init_action_mode == "full_candidate":
            # pick action with max number of APs
            lengths = [int(np.sum(a)) for a in actions_k]
            idx = int(np.argmax(lengths))

        elif init_action_mode == "random":
            idx = int(rng.integers(0, len(actions_k)))

        elif init_action_mode == "strongest_singleton":
            candidate_aps = graph.candidate_aps_per_user[k]
            if candidate_aps.size == 0:
                idx = 0
            else:
                tr_ru_k = np.asarray(pilot.tr_Ru[s, :, k], dtype=float)
                best_ap = int(candidate_aps[np.argmax(tr_ru_k[candidate_aps])])

                idx = 0
                for j, a in enumerate(actions_k):
                    if np.sum(a) == 1 and bool(a[best_ap]):
                        idx = j
                        break
        else:
            raise ValueError(f"Unknown init_action_mode {init_action_mode}")

        action_masks.append(actions_k[idx].copy())

    return action_masks


def solve_association_game_fixed_alpha_one_setup(
    cfg: Config,
    geom: GeometrySetups,
    ls: LargeScaleSetups,
    corr: CorrelationSetups,
    phases: PhaseShiftSetups,
    pilot: PilotAssignmentSetups,
    est_stats: ChannelEstimationSetups,
    graph: StaticGameGraphSetup,
    *,
    s: int,
    alpha: float,
    options: GameOptions,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    initial_serve_mask: Optional[np.ndarray] = None,
) -> FixedAlphaGameSetupResult:
    """
    Solve the local altruistic association game for one setup at fixed alpha.

    Default behavior:
    - deterministic best response,
    - accept only utility-improving actions,
    - track the incumbent profile with the best utility proxy,
    - stop on utility stability, no changed users, or detected cycles.
    """
    options.validate()

    rng = np.random.default_rng(options.random_seed)

    cfg1 = _single_setup_cfg(cfg)
    ls1 = _slice_large_scale(ls, s)
    corr1 = _slice_corr(corr, s)
    phases1 = _slice_phases(phases, s)
    est1 = _slice_est_stats(est_stats, s)
    pilot_of_user1 = _slice_pilot_of_user(pilot.pilot_of_user, s)

    if initial_serve_mask is None:
        current_actions = _initialize_actions_for_setup(
            cfg=cfg,
            graph=graph,
            pilot=pilot,
            s=s,
            init_action_mode=options.init_action_mode,
            rng=rng,
        )
    else:
        current_actions = _actions_from_serve_mask(graph, initial_serve_mask)

    transformed_objective_history: list[float] = []
    utility_proxy_history: list[float] = []
    ee_bit_per_joule_history: list[float] = []
    sum_rate_bps_history: list[float] = []
    total_power_watt_history: list[float] = []
    chosen_set_history: list[int] = []
    num_users_changed_history: list[int] = []

    current_mask = _actions_to_serve_mask(current_actions)
    initial_mask = current_mask.copy()

    if options.enforce_user_served:
        _validate_all_users_served(current_mask, where=f"setup={s} initial profile")

    power_out, sinr_cf, se_out, ee_out, _ = _evaluate_profile_global(
        cfg=cfg1,
        est_stats=est1,
        corr=corr1,
        ls=ls1,
        phases=phases1,
        pilot_of_user=pilot_of_user1,
        serve_mask=current_mask,
        redistribution_scheme=options.redistribution_scheme,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    current_obj = _global_transformed_objective(alpha, ee_out)
    current_utility_proxy = _profile_utility_proxy(
        cfg=cfg1,
        graph=graph,
        alpha=alpha,
        serve_mask=current_mask,
        power_out=power_out,
        se_out=se_out,
        ee_out=ee_out,
    )

    current_ee_bit_per_joule = float(ee_out.ee_bit_per_joule[0])
    current_sum_rate_bps = float(ee_out.sum_rate_bps[0])
    current_total_power_watt = float(ee_out.total_power_watt[0])

    transformed_objective_history.append(current_obj)
    utility_proxy_history.append(current_utility_proxy)
    ee_bit_per_joule_history.append(current_ee_bit_per_joule)
    sum_rate_bps_history.append(current_sum_rate_bps)
    total_power_watt_history.append(current_total_power_watt)

    # Incumbent according to the local utility proxy.
    best_mask = current_mask.copy()
    best_power = power_out
    best_sinr = sinr_cf
    best_se = se_out
    best_ee = ee_out
    best_utility_proxy = current_utility_proxy

    # Diagnostic only: best mask according to global EE.
    best_global_ee_mask = current_mask.copy()
    best_global_ee_bit_per_joule = current_ee_bit_per_joule

    seen_masks = {_mask_key(current_mask)} if options.cycle_detection else set()
    converged = False
    cycle_detected = False

    for t in range(1, options.max_game_iters + 1):
        set_idx = int(rng.integers(0, len(graph.maximum_non_neighbor_sets)))
        chosen_set = graph.maximum_non_neighbor_sets[set_idx]
        chosen_set_history.append(set_idx)

        old_actions = [a.copy() for a in current_actions]
        proposed_updates: dict[int, np.ndarray] = {}

        omega_t = _omega_schedule(t, options.omega_initial)

        for k in chosen_set:
            qk = graph.q_sets[k]
            actions_k = graph.action_masks_per_user[k]
            utilities_k = np.full(len(actions_k), -np.inf, dtype=float)

            current_idx = _find_action_index(actions_k, old_actions[int(k)])
            if current_idx is None:
                current_idx = 0

            for a_idx, action_mask_k in enumerate(actions_k):
                trial_actions = [a.copy() for a in old_actions]
                trial_actions[int(k)] = action_mask_k.copy()
                trial_mask = _actions_to_serve_mask(trial_actions)

                if options.enforce_user_served and not _all_users_served(trial_mask):
                    continue

                p_trial, sinr_trial, se_trial, ee_trial, _ = _evaluate_profile_global(
                    cfg=cfg1,
                    est_stats=est1,
                    corr=corr1,
                    ls=ls1,
                    phases=phases1,
                    pilot_of_user=pilot_of_user1,
                    serve_mask=trial_mask,
                    redistribution_scheme=options.redistribution_scheme,
                    pilot_power_watt_per_user=pilot_power_watt_per_user,
                )

                utilities_k[a_idx] = _evaluate_local_utility(
                    cfg=cfg1,
                    q_set_k=qk,
                    alpha=alpha,
                    serve_mask=trial_mask,
                    power_out=p_trial,
                    se_out=se_trial,
                    ee_out=ee_trial,
                )

            finite_idx = np.where(np.isfinite(utilities_k))[0]
            chosen_action_idx = int(current_idx)

            if finite_idx.size > 0:
                if options.response_mode == "best_response":
                    best_local_idx = int(finite_idx[np.argmax(utilities_k[finite_idx])])
                    current_utility = utilities_k[current_idx]
                    best_utility = utilities_k[best_local_idx]

                    if (not np.isfinite(current_utility)) or (
                        best_utility > current_utility + options.utility_improvement_tol
                    ):
                        chosen_action_idx = best_local_idx
                else:
                    sampled_idx = _sample_logit_action(utilities_k[finite_idx], omega_t, rng)
                    chosen_action_idx = int(finite_idx[sampled_idx])

            proposed_updates[int(k)] = actions_k[chosen_action_idx].copy()

        num_changed = 0
        for k, action_mask_k in proposed_updates.items():
            if not np.array_equal(current_actions[k], action_mask_k):
                num_changed += 1
            current_actions[k] = action_mask_k
        num_users_changed_history.append(int(num_changed))

        current_mask = _actions_to_serve_mask(current_actions)
        if options.enforce_user_served:
            _validate_all_users_served(current_mask, where=f"setup={s} game_iter={t}")

        power_out, sinr_cf, se_out, ee_out, _ = _evaluate_profile_global(
            cfg=cfg1,
            est_stats=est1,
            corr=corr1,
            ls=ls1,
            phases=phases1,
            pilot_of_user=pilot_of_user1,
            serve_mask=current_mask,
            redistribution_scheme=options.redistribution_scheme,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
        )
        current_obj = _global_transformed_objective(alpha, ee_out)
        current_utility_proxy = _profile_utility_proxy(
            cfg=cfg1,
            graph=graph,
            alpha=alpha,
            serve_mask=current_mask,
            power_out=power_out,
            se_out=se_out,
            ee_out=ee_out,
        )

        current_ee_bit_per_joule = float(ee_out.ee_bit_per_joule[0])
        current_sum_rate_bps = float(ee_out.sum_rate_bps[0])
        current_total_power_watt = float(ee_out.total_power_watt[0])

        transformed_objective_history.append(current_obj)
        utility_proxy_history.append(current_utility_proxy)
        ee_bit_per_joule_history.append(current_ee_bit_per_joule)
        sum_rate_bps_history.append(current_sum_rate_bps)
        total_power_watt_history.append(current_total_power_watt)

        if current_utility_proxy > best_utility_proxy + options.utility_improvement_tol:
            best_utility_proxy = current_utility_proxy
            best_mask = current_mask.copy()
            best_power = power_out
            best_sinr = sinr_cf
            best_se = se_out
            best_ee = ee_out

        if current_ee_bit_per_joule > best_global_ee_bit_per_joule + 1e-12:
            best_global_ee_bit_per_joule = current_ee_bit_per_joule
            best_global_ee_mask = current_mask.copy()

        if num_changed == 0:
            converged = True
            break

        if options.cycle_detection:
            key = _mask_key(current_mask)
            if key in seen_masks:
                converged = True
                cycle_detected = True
                break
            seen_masks.add(key)

        if len(utility_proxy_history) >= options.convergence_window + 1:
            recent = np.asarray(
                utility_proxy_history[-(options.convergence_window + 1):],
                dtype=float,
            )
            rel_changes = np.array(
                [_relative_change(recent[i], recent[i - 1]) for i in range(1, recent.size)],
                dtype=float,
            )
            if np.all(rel_changes <= options.stability_tol):
                converged = True
                break

    terminal_mask = current_mask.copy()

    if options.incumbent_tracking:
        current_mask = best_mask
        power_out = best_power
        sinr_cf = best_sinr
        se_out = best_se
        ee_out = best_ee

    return FixedAlphaGameSetupResult(
        initial_serve_mask=initial_mask,
        final_serve_mask=current_mask,
        terminal_serve_mask=terminal_mask,
        best_utility_serve_mask=best_mask,
        best_global_ee_serve_mask=best_global_ee_mask,
        final_power=power_out,
        final_sinr_cf=sinr_cf,
        final_se=se_out,
        final_ee=ee_out,
        transformed_objective_history=transformed_objective_history,
        utility_proxy_history=utility_proxy_history,
        ee_bit_per_joule_history=ee_bit_per_joule_history,
        sum_rate_bps_history=sum_rate_bps_history,
        total_power_watt_history=total_power_watt_history,
        chosen_set_history=chosen_set_history,
        num_users_changed_history=num_users_changed_history,
        best_utility_proxy=float(best_utility_proxy),
        best_global_ee_bit_per_joule=float(best_global_ee_bit_per_joule),
        converged=converged,
        num_iterations=len(transformed_objective_history) - 1,
        cycle_detected=cycle_detected,
    )


# ---------------------------------------------------------------------
# Outer Dinkelbach optimizer
# ---------------------------------------------------------------------

def optimize_association_game_ee(
    cfg: Config,
    geom: GeometrySetups,
    ls: LargeScaleSetups,
    corr: CorrelationSetups,
    phases: PhaseShiftSetups,
    pilot: PilotAssignmentSetups,
    *,
    game_options: Optional[GameOptions] = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
) -> AssociationOptimizerResults:
    """
    Optimize AP-user association for energy efficiency using

    - static-neighbor local altruistic game at fixed alpha,
    - deterministic best-response updates by default,
    - utility-proxy incumbent tracking,
    - optional warm start between outer Dinkelbach iterations,
    - equal-power redistribution induced by each serving mask.

    Important design rule
    - No optimized-power call is allowed inside this file.
    - Every serving-mask evaluation uses equal-power redistribution.
    """
    cfg.validate()

    options = GameOptions(d_th=100.0) if game_options is None else game_options
    options.validate()

    S = int(cfg.sim.num_setups)
    rng = np.random.default_rng(options.random_seed)

    graphs = build_static_game_graphs(
        cfg=cfg,
        geom=geom,
        pilot=pilot,
        d_th=options.d_th,
        max_subset_size=options.max_subset_size,
        max_actions_per_user=options.max_actions_per_user,
    )

    est_stats_all = build_channel_estimation_statistics(
        cfg=cfg,
        corr=corr,
        ls=ls,
        pilot_of_user=pilot.pilot_of_user,
        pilot_groups=pilot.pilot_groups,
        phases=phases,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    fixed_alpha_results_by_outer_iter: list[list[FixedAlphaGameSetupResult]] = []
    alpha_history: list[float] = []
    residual_history: list[float] = []
    outer_mean_ee_history: list[float] = []
    outer_mean_se_history: list[float] = []
    outer_avg_serving_aps_per_user_history: list[float] = []
    outer_num_users_with_mask_change_history: list[int] = []

    alpha = float(options.alpha_init)
    converged = False
    previous_final_mask: Optional[np.ndarray] = None
    previous_outer_ee: Optional[float] = None
    warm_start_mask: Optional[np.ndarray] = None

    final_mask: Optional[np.ndarray] = None
    power_final: Optional[PowerAllocationSetups] = None
    sinr_final: Optional[DownlinkSinrClosedFormSetups] = None
    se_final: Optional[SpectralEfficiencySetups] = None
    ee_final: Optional[EnergyEfficiencySetups] = None

    for outer_idx in range(options.max_dinkelbach_iters):
        alpha_history.append(float(alpha))
        outer_results: list[FixedAlphaGameSetupResult] = []

        for s in range(S):
            init_mask_s = None
            if options.outer_warm_start and warm_start_mask is not None:
                init_mask_s = warm_start_mask[s:s + 1]

            result_s = solve_association_game_fixed_alpha_one_setup(
                cfg=cfg,
                geom=geom,
                ls=ls,
                corr=corr,
                phases=phases,
                pilot=pilot,
                est_stats=est_stats_all,
                graph=graphs[s],
                s=s,
                alpha=alpha,
                options=replace(options, random_seed=int(rng.integers(0, 2**31 - 1))),
                pilot_power_watt_per_user=pilot_power_watt_per_user,
                initial_serve_mask=init_mask_s,
            )
            outer_results.append(result_s)

        fixed_alpha_results_by_outer_iter.append(outer_results)

        final_mask = np.concatenate([r.final_serve_mask for r in outer_results], axis=0)
        if options.enforce_user_served:
            _validate_all_users_served(final_mask, where=f"outer_iter={outer_idx} final mask")

        power_final = allocate_downlink_power(
            cfg,
            scheme="equal",
            serve_mask=final_mask,
        )

        sinr_final = compute_dl_sinr_closed_form(
            cfg=cfg,
            est=est_stats_all,
            corr=corr,
            ls=ls,
            power=power_final,
            pilot_of_user=pilot.pilot_of_user,
            phases=phases,
            serve_mask=final_mask,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
        )

        se_final = compute_spectral_efficiency_setups(cfg, sinr_final.sinr_dl)
        ee_final = compute_energy_efficiency_setups(cfg, se_final, power_final, serve_mask=final_mask)

        numerator = float(np.mean(ee_final.sum_rate_bps))
        denominator = float(np.mean(ee_final.total_power_watt))

        residual = numerator - alpha * denominator
        relative_residual = abs(residual) / max(abs(numerator), 1e-12)
        residual_history.append(relative_residual)

        mean_ee_value = float(np.mean(ee_final.ee_bit_per_joule))
        aggregate_ee_value = numerator / max(denominator, 1e-12)

        outer_mean_ee_history.append(mean_ee_value)
        outer_mean_se_history.append(float(np.mean(se_final.sum_se)))
        outer_avg_serving_aps_per_user_history.append(float(np.mean(np.sum(final_mask, axis=1))))

        if previous_final_mask is None:
            num_users_changed = 0
        else:
            changed = np.any(previous_final_mask != final_mask, axis=1)  # (S, K)
            num_users_changed = int(np.sum(changed))
        outer_num_users_with_mask_change_history.append(num_users_changed)
        previous_final_mask = final_mask.copy()
        warm_start_mask = final_mask.copy()

        alpha_new = aggregate_ee_value

        if previous_outer_ee is not None:
            outer_relative_ee_change = _relative_change(aggregate_ee_value, previous_outer_ee)
            if outer_relative_ee_change <= options.dinkelbach_tol:
                converged = True
                alpha = alpha_new
                break

        previous_outer_ee = aggregate_ee_value
        alpha = alpha_new

    if final_mask is None or power_final is None or sinr_final is None or se_final is None or ee_final is None:
        raise RuntimeError("Association optimizer finished without producing final outputs")

    return AssociationOptimizerResults(
        final_serve_mask=final_mask,
        final_power=power_final,
        final_sinr_cf=sinr_final,
        final_se=se_final,
        final_ee=ee_final,
        graphs=graphs,
        fixed_alpha_results_by_outer_iter=fixed_alpha_results_by_outer_iter,
        alpha_history=alpha_history,
        residual_history=residual_history,
        outer_mean_ee_history=outer_mean_ee_history,
        outer_mean_se_history=outer_mean_se_history,
        outer_avg_serving_aps_per_user_history=outer_avg_serving_aps_per_user_history,
        outer_num_users_with_mask_change_history=outer_num_users_with_mask_change_history,
        converged=converged,
        num_outer_iterations=len(alpha_history),
        options=options,
    )


def optimize_association_and_power_ee(
    cfg: Config,
    geom: GeometrySetups,
    ls: LargeScaleSetups,
    corr: CorrelationSetups,
    phases: PhaseShiftSetups,
    pilot: PilotAssignmentSetups,
    *,
    d_th: float,
    power_scheme: str = "equal",
    association_eta_mode: str = "recompute_from_scheme",
    eta_fixed_reference: Optional[np.ndarray] = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    init_action_mode: InitActionMode = "strongest_singleton",
    response_mode: ResponseMode = "best_response",
    omega_initial: float = 5.0,
    max_game_iters: int = 50,
    max_dinkelbach_iters: int = 10,
    dinkelbach_tol: float = 1e-4,
    convergence_window: int = 5,
    stability_tol: float = 1e-5,
    utility_improvement_tol: float = 1e-12,
    incumbent_tracking: bool = True,
    outer_warm_start: bool = True,
    cycle_detection: bool = True,
    max_subset_size: Optional[int] = None,
    max_actions_per_user: Optional[int] = None,
    random_seed: Optional[int] = None,
    optimized_power_options: object | None = None,
) -> AssociationOptimizerResults:
    """
    Backward-compatible wrapper.

    Updated design policy
    - The optimizer keeps Dinkelbach iterations.
    - Inside this file, every mask is evaluated using equal-power redistribution.
    - No fixed-power-reference mode or optimized-power call is allowed here anymore.
    """
    if power_scheme != "equal":
        raise ValueError(
            "association_game_optimizer.py now supports only equal-power redistribution internally. "
            "Use the power optimizer outside this file after the final mask is obtained."
        )
    if association_eta_mode != "recompute_from_scheme":
        raise ValueError(
            "association_game_optimizer.py no longer supports fixed-reference eta mode. "
            "All mask evaluations use equal-power redistribution induced by the current mask."
        )
    if eta_fixed_reference is not None:
        raise ValueError(
            "eta_fixed_reference is no longer used in association_game_optimizer.py. "
            "Run the game with equal redistribution here, then call power optimization outside if needed."
        )
    if optimized_power_options is not None:
        raise ValueError(
            "optimized_power_options is no longer used in association_game_optimizer.py. "
            "Call the optimized power allocator outside this file after association is finalized."
        )

    options = GameOptions(
        d_th=d_th,
        redistribution_scheme="equal",
        init_action_mode=init_action_mode,
        response_mode=response_mode,
        omega_initial=omega_initial,
        max_game_iters=max_game_iters,
        max_dinkelbach_iters=max_dinkelbach_iters,
        dinkelbach_tol=dinkelbach_tol,
        convergence_window=convergence_window,
        stability_tol=stability_tol,
        utility_improvement_tol=utility_improvement_tol,
        incumbent_tracking=incumbent_tracking,
        outer_warm_start=outer_warm_start,
        cycle_detection=cycle_detection,
        max_subset_size=max_subset_size,
        max_actions_per_user=max_actions_per_user,
        random_seed=random_seed,
        alpha_init=0.0,
        enforce_user_served=True,
    )
    return optimize_association_game_ee(
        cfg=cfg,
        geom=geom,
        ls=ls,
        corr=corr,
        phases=phases,
        pilot=pilot,
        game_options=options,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )