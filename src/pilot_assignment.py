# ITNG YMYFA 80asra YA
# IMZZ

# src/pilot_assignment.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .config import Config
from .geometry import GeometrySetups, distances_2d
from .large_scale import LargeScaleSetups
from .correlation_setups import CorrelationSetups
from .ris_phase_shifts import PhaseShiftSetups


PilotScheme = Literal["random", "heuristic"]


@dataclass(frozen=True)
class PilotAssignmentSetups:
    """
    Pilot assignment outputs for all setups.

    Shapes
    - pilot_of_user: (S, K) values in [0, tau_p-1]
    - master_ap: (S, K) values in [0, L-1]
    - candidate_mask: (S, L, K) bool, True means AP l is in L_k^c
    - pilot_group_mask: (S, tau_p, K) bool, True means user k is in P_t

    Lists
    - pilot_groups[s][t] is an array of user indices using pilot t in setup s
    - candidate_aps_per_user[s][k] is an array of AP indices in L_k^c for setup s

    Debug
    - served_user_on_pilot: (S, L, tau_p) stores which user is served by AP l on pilot t, -1 if empty
    - tr_Ru: (S, L, K) stores tr(R_u_lk) used by the heuristic
    - chi: (S,) chi values per setup, 0 when RIS is disabled
    """
    pilot_of_user: np.ndarray
    master_ap: np.ndarray
    candidate_mask: np.ndarray
    pilot_group_mask: np.ndarray

    pilot_groups: list[list[np.ndarray]]
    candidate_aps_per_user: list[list[np.ndarray]]

    served_user_on_pilot: np.ndarray
    tr_Ru: np.ndarray
    chi: np.ndarray



def _compute_neighbor_sets(ap_xy: np.ndarray, cfg: Config, Q: int) -> np.ndarray:
    """
    Build M_l neighbor sets using Q-nearest rule.

    ap_xy shape (L, 2)
    return shape (L, Q+1)
    """
    L = ap_xy.shape[0]
    Q_eff = int(min(max(Q, 0), L - 1))

    D = distances_2d(ap_xy[:, None, :], ap_xy[None, :, :], cfg)     # (L, L)
    idx = np.argsort(D, axis=1)[:, : Q_eff + 1]                     # includes self at index 0
    return idx.astype(int, copy=False)



def _compute_chi_per_setup(
        cfg: Config,
        corr: CorrelationSetups,
        phases: Optional[PhaseShiftSetups],
) -> np.ndarray:
    """
    Compute chi per setup

      chi = tr(Theta R Theta^H R)

    chi is user-independent by your design and depends only on setup-level Theta.

    Returns chi with shape (S,)
    """
    S = int(cfg.sim.num_setups)

    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        return np.zeros(S, dtype=float)

    if phases is None:
        raise ValueError("phases must be provided when RIS is enabled")

    R = np.asarray(corr.R_ris)
    if R.size == 0:
        return np.zeros(S, dtype=float)

    vartheta = np.asarray(phases.vartheta, dtype=float)
    if vartheta.shape[0] != S:
        raise ValueError("phases.vartheta must have first dimension equal to num_setups")

    chi = np.zeros(S, dtype=float)

    for s in range(S):
        d = np.exp(1j * vartheta[s])  # (N,)
        X = (d[:, None] * R) * np.conj(d[None, :])
        val = np.sum(X * R.T)
        chi[s] = float(np.real_if_close(val))

    return chi



def _compute_tr_Ru(
    cfg: Config,
    ls: LargeScaleSetups,
    chi: np.ndarray,
) -> np.ndarray:
    """
    Compute tr(R_u_lk) for every setup, AP, user.

    Using trace-normalized Delta where tr(Delta)=M, we use

      tr(R_u_lk) = tr(R_f_lk) + hat_beta_l * tilde_beta_k * chi * tr(Delta)

    In this pipeline we use noise-normalized betas and set sigma_ul^2 = 1.
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = float(cfg.dims.num_ap_antennas)

    direct = M * np.asarray(ls.ap_user_beta_over_noise, dtype=float)  # (S, L, K)

    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        return direct

    ap_ris = np.asarray(ls.ap_ris_beta_over_noise, dtype=float)       # (S, L, T)
    ris_user = np.asarray(ls.ris_user_beta_over_noise, dtype=float)   # (S, T, K)

    if ap_ris.size == 0 or ris_user.size == 0:
        return direct

    casc = np.zeros((S, L, K), dtype=float)

    # General form supports T >= 1
    # casc[s,l,k] = chi[s] * M * sum_t ap_ris[s,l,t] * ris_user[s,t,k]
    for s in range(S):
        casc[s] = (chi[s] * M) * (ap_ris[s] @ ris_user[s])  # (L, K)

    return direct + casc



def assign_pilots_setups(
    cfg: Config,
    geom: GeometrySetups,
    ls: LargeScaleSetups,
    corr: CorrelationSetups,
    phases: Optional[PhaseShiftSetups],
    scheme: PilotScheme = "heuristic",
    Q: int = 5,
    protect_masters: bool = True,
    seed: Optional[int] = None,
) -> PilotAssignmentSetups:
    """
    Pilot assignment with two schemes.
    - random picks a pilot uniformly
    - heuristic follows the argmin tr(Omega) rule at the master AP

    Uses noise-normalized betas and sets sigma_ul^2 = 1 so tr(sigma_ul^2 I_M) = M.
    """
    cfg.validate()

    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    tau_p = int(round(float(cfg.tau_p())))
    if tau_p <= 0:
        raise ValueError("tau_p must be positive.")
    
    p_pilot = float(cfg.pilots.pilot_power_watt)
    M = float(cfg.dims.num_ap_antennas)

    chi = _compute_chi_per_setup(cfg, corr, phases)         # (S,)
    tr_Ru = _compute_tr_Ru(cfg, ls, chi)                    # (S, L, K)

    pilot_of_user = -np.ones((S, K), dtype=int)
    master_ap = -np.ones((S, K), dtype=int)

    candidate_mask = np.zeros((S, L, K), dtype=bool)
    pilot_group_mask = np.zeros((S, tau_p, K), dtype=bool)

    served_user_on_pilot = -np.ones((S, L, tau_p), dtype=int)

    pilot_groups_all: list[list[np.ndarray]] = []
    candidate_aps_all: list[list[np.ndarray]] = []

    for s in range(S):
        ap_xy = np.asarray(geom.ap_xy[s], dtype=float)      # (L, 2)
        M_l = _compute_neighbor_sets(ap_xy, cfg, Q=Q)       # (L, Q+1)

        groups: list[list[int]] = [[] for _ in range(tau_p)]
        sum_trace_ru = np.zeros((L, tau_p), dtype=float)

        # master flags per AP and pilot
        is_master_slot = np.zeros((L, tau_p), dtype=bool)

        # current load per AP
        ap_load = np.zeros(L, dtype=int)

        for k in range(K):
            # Step 1, master AP selection among not-full APs
            scores = tr_Ru[s, :, k]  # (L,)
            not_full = ap_load < tau_p

            if np.any(not_full):
                l_star = int(np.argmax(np.where(not_full, scores, -np.inf)))
            else:
                l_star = int(np.argmax(scores))

            master_ap[s, k] = l_star

            # Step 2, pilot selection at master AP
            allowed = np.ones(tau_p, dtype=bool)

            # exclude pilots where master AP is already master for another user
            allowed &= ~is_master_slot[l_star]

            # prefer empty pilots at master AP if any exist
            empty_here = served_user_on_pilot[s, l_star] == -1
            if np.any(allowed & empty_here):
                allowed = allowed & empty_here

            if scheme == "random":
                choices = np.where(allowed)[0]
                if choices.size == 0:
                    choices = np.arange(tau_p)
                t_k = int(rng.choice(choices))
            elif scheme == "heuristic":
                tr_Omega = p_pilot * tau_p * sum_trace_ru[l_star] + M
                candidates = np.where(allowed)[0]
                if candidates.size == 0:
                    candidates = np.arange(tau_p)
                t_k = int(candidates[np.argmin(tr_Omega[candidates])])
            else:
                raise ValueError(f"Unknown scheme {scheme}")
            
            pilot_of_user[s, k] = t_k
            pilot_group_mask[s, t_k, k] = True
            groups[t_k].append(k)

            # Update pilot contamination sum for every AP for this pilot
            sum_trace_ru[:, t_k] += tr_Ru[s, :, k]

            # Step 3, pre-cluster formation in M_{l_star}
            neighbors = M_l[l_star]

            for l in neighbors:
                l = int(l)
                current = int(served_user_on_pilot[s, l, t_k])

                if current == -1:
                    served_user_on_pilot[s, l, t_k] = k
                    candidate_mask[s, l, k] = True
                    ap_load[l] += 1

                    if l == l_star:
                        is_master_slot[l, t_k] = True

                else:
                    if protect_masters and is_master_slot[l, t_k]:
                        continue

                    if tr_Ru[s, l, k] > tr_Ru[s, l, current]:
                        served_user_on_pilot[s, l, t_k] = k
                        candidate_mask[s, l, k] = True
                        candidate_mask[s, l, current] = False

            # Enforce master AP must serve user k
            current_master_slot = int(served_user_on_pilot[s, l_star, t_k])
            if current_master_slot != k:
                old = current_master_slot

                if old == -1:
                    ap_load[l_star] += 1
                else:
                    candidate_mask[s, l_star, old] = False

                served_user_on_pilot[s, l_star, t_k] = k
                candidate_mask[s, l_star, k] = True
                is_master_slot[l_star, t_k] = True

        pilot_groups_all.append([np.array(g, dtype=int) for g in groups])

        cand_per_user: list[np.ndarray] = []
        for k in range(K):
            cand_per_user.append(np.where(candidate_mask[s, :, k])[0].astype(int))
        candidate_aps_all.append(cand_per_user)

    return PilotAssignmentSetups(
        pilot_of_user=pilot_of_user,
        master_ap=master_ap,
        candidate_mask=candidate_mask,
        pilot_group_mask=pilot_group_mask,
        pilot_groups=pilot_groups_all,
        candidate_aps_per_user=candidate_aps_all,
        served_user_on_pilot=served_user_on_pilot,
        tr_Ru=tr_Ru,
        chi=chi,
    )