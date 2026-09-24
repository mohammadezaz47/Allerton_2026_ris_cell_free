# ITNG YMYFA 80asra YA
# IMZZ

# src/channel_estimation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import Config
from .large_scale import LargeScaleSetups
from .correlation_setups import CorrelationSetups
from .channels import ChannelSetups
from .ris_phase_shifts import PhaseShiftSetups


@dataclass(frozen=True)
class ChannelEstimationSetups:
    """
    MMSE channel estimation outputs.

    This object is used in two modes.

    Statistics-only mode
    - u_hat is None
    - enough for closed-form SINR

    Full mode
    - u_hat is present with shape (S, L, M, K, R)
    - enough for Monte Carlo SINR

    Main statistical outputs
    - R_u:       (S, L, K, M, M)
    - Omega:     (S, L, tau_p, M, M)
    - Omega_inv: (S, L, tau_p, M, M)
    - W:         (S, L, K, M, M)
    - R_hat:     (S, L, K, M, M)
    - C:         (S, L, K, M, M)

    Realization-level output
    - u_hat:     (S, L, M, K, R) or None
    """
    u_hat: np.ndarray | None

    R_u: np.ndarray
    Omega: np.ndarray
    Omega_inv: np.ndarray
    W: np.ndarray
    R_hat: np.ndarray
    C: np.ndarray

    pilot_of_user: np.ndarray
    chi: np.ndarray

    tau_p: int
    pilot_power_watt_per_user: np.ndarray  # (K,)


def _complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator, dtype: np.dtype) -> np.ndarray:
    """
    i.i.d. CN(0,1) samples.
    """
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    out = (real + 1j * imag) / np.sqrt(2.0)
    return out.astype(dtype, copy=False)


def _pilot_groups_from_indices(pilot_of_user: np.ndarray, tau_p: int) -> list[list[np.ndarray]]:
    """
    Build pilot groups from pilot indices.

    Returns
    - groups[s][t] = array of user indices using pilot t
    """
    S, K = pilot_of_user.shape
    groups: list[list[np.ndarray]] = []

    for s in range(S):
        groups_s: list[np.ndarray] = []
        for t in range(tau_p):
            users_t = np.where(pilot_of_user[s] == t)[0].astype(int)
            groups_s.append(users_t)
        groups.append(groups_s)

    return groups


def _compute_chi_per_setup(
    cfg: Config,
    corr: CorrelationSetups,
    phases: Optional[PhaseShiftSetups],
) -> np.ndarray:
    """
    Compute chi per setup

      chi = tr(Theta R Theta^H R)

    Theta is diagonal with entries exp(j*vartheta_n).
    chi is user-independent by your design.

    Returns chi with shape (S,)
    """
    S = int(cfg.sim.num_setups)

    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        return np.zeros(S, dtype=float)

    if phases is None:
        raise ValueError("phases must be provided when RIS is enabled")

    R = np.asarray(corr.R_ris, dtype=np.complex128)
    if R.size == 0:
        return np.zeros(S, dtype=float)

    vartheta = np.asarray(phases.vartheta, dtype=float)
    if vartheta.shape[0] != S:
        raise ValueError("phases.vartheta must have first dimension equal to num_setups")

    chi = np.zeros(S, dtype=float)

    for s in range(S):
        d = np.exp(1j * vartheta[s])  # (N,)
        X = (d[:, None] * R) * np.conj(d[None, :])  # Theta R Theta^H
        val = np.sum(X * R.T)  # tr(XR)
        chi[s] = float(np.real_if_close(val))

    return chi


def _build_R_u_all_links(
    cfg: Config,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    chi: np.ndarray,
) -> np.ndarray:
    """
    Build R_u for all (s,l,k).

      R_u_lk = R_f_lk + (sum_t hat_beta_{l,t} * tilde_beta_{t,k}) * chi * Delta

    Here
    - R_f_lk is the direct-link covariance, we use corr.R_ap_user
    - hat_beta is ls.ap_ris_beta_over_noise
    - tilde_beta is ls.ris_user_beta_over_noise
    - Delta is taken from corr.delta_ap_ris
    - In no-RIS mode, the RIS term is automatically zero

    Output shape
    - (S, L, K, M, M)
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)

    R_u = np.asarray(corr.R_ap_user, dtype=np.complex128).copy()  # (S, L, K, M, M)

    T = int(corr.delta_ap_ris.shape[2]) if corr.delta_ap_ris.ndim >= 3 else 0
    ris_active = (cfg.ris.enable_ris and T > 0 and ls.ap_ris_beta_over_noise.size > 0)

    if not ris_active:
        R_u = 0.5 * (R_u + np.swapaxes(R_u.conj(), -1, -2))
        return R_u

    ap_ris = np.asarray(ls.ap_ris_beta_over_noise, dtype=float)      # (S, L, T)
    ris_user = np.asarray(ls.ris_user_beta_over_noise, dtype=float)  # (S, T, K)

    # Use corr.delta_ap_ris as requested. With your master-Delta design, this is shared.
    Delta = np.asarray(corr.delta_ap_ris[..., 0, :, :], dtype=np.complex128)  # (S, L, M, M)

    for s in range(S):
        beta_casc = ap_ris[s] @ ris_user[s]  # (L, K)

        for l in range(L):
            R_u[s, l] += (chi[s] * beta_casc[l])[:, None, None] * Delta[s, l][None, :, :]

    R_u = 0.5 * (R_u + np.swapaxes(R_u.conj(), -1, -2))
    return R_u


def _invert_hermitian_stack(A: np.ndarray) -> np.ndarray:
    """
    Invert a stack of small Hermitian matrices.

    Input shape (..., M, M)
    Output shape (..., M, M)
    """
    A = np.asarray(A, dtype=np.complex128)
    I = np.eye(A.shape[-1], dtype=np.complex128)
    out = np.empty_like(A)

    flat = A.reshape(-1, A.shape[-2], A.shape[-1])
    out_flat = out.reshape(-1, A.shape[-2], A.shape[-1])

    for i in range(flat.shape[0]):
        out_flat[i] = np.linalg.solve(flat[i], I)

    return out


def build_channel_estimation_statistics(
    cfg: Config,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    pilot_of_user: np.ndarray,
    pilot_groups: Optional[Sequence[Sequence[np.ndarray]]] = None,
    phases: Optional[PhaseShiftSetups] = None,
    *,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
) -> ChannelEstimationSetups:
    """
    Build only the statistical objects needed for MMSE estimation.

    This is enough for the closed-form SINR branch and does not require any channel realizations.

    Outputs
    - R_u
    - Omega
    - Omega_inv
    - W
    - R_hat
    - C
    - chi
    - u_hat is None
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)

    tau_p = int(cfg.tau_p())
    if tau_p <= 0:
        raise ValueError("tau_p must be positive")

    pilot_of_user = np.asarray(pilot_of_user, dtype=int)
    if pilot_of_user.shape != (S, K):
        raise ValueError("pilot_of_user must have shape (S, K)")

    if pilot_groups is None:
        pilot_groups_eff = _pilot_groups_from_indices(pilot_of_user, tau_p)
    else:
        if len(pilot_groups) != S:
            raise ValueError("pilot_groups must have length S")
        pilot_groups_eff = [list(g) for g in pilot_groups]

    # Pilot powers
    if pilot_power_watt_per_user is None:
        p_vec = np.full(K, float(cfg.pilots.pilot_power_watt), dtype=float)
    else:
        p_vec = np.asarray(pilot_power_watt_per_user, dtype=float).reshape(-1)
        if p_vec.shape[0] != K:
            raise ValueError("pilot_power_watt_per_user must have length K")
        if np.any(p_vec < 0):
            raise ValueError("pilot powers must be non-negative")

    # 1) chi per setup
    chi = _compute_chi_per_setup(cfg, corr, phases)  # (S,)

    # 2) Covariance of u for all links
    R_u = _build_R_u_all_links(cfg, corr, ls, chi)  # (S, L, K, M, M)

    # 3) Build Omega for all (s,l,t)
    Omega = np.zeros((S, L, tau_p, M, M), dtype=np.complex128)
    I_M = np.eye(M, dtype=np.complex128)

    for s in range(S):
        for t in range(tau_p):
            users_t = np.asarray(pilot_groups_eff[s][t], dtype=int).ravel()

            if users_t.size == 0:
                Omega[s, :, t] = I_M
                continue

            weights = (p_vec[users_t] * tau_p).astype(float)  # (|P_t|,)
            for l in range(L):
                Ru_sum = np.zeros((M, M), dtype=np.complex128)
                for idx, i in enumerate(users_t):
                    Ru_sum += weights[idx] * R_u[s, l, i]
                Omega[s, l, t] = 0.5 * (Ru_sum + I_M + (Ru_sum + I_M).conj().T) / 1.0

    # The previous line double-counted the identity if written carelessly.
    # So fix Omega explicitly to the correct Hermitian version:
    for s in range(S):
        for l in range(L):
            for t in range(tau_p):
                A = Omega[s, l, t]
                Omega[s, l, t] = 0.5 * (A + A.conj().T)

    # 4) Invert Omega
    Omega_inv = _invert_hermitian_stack(Omega)  # (S, L, tau_p, M, M)

    # 5) Build W, R_hat, C for all links
    W = np.zeros((S, L, K, M, M), dtype=np.complex128)
    R_hat = np.zeros((S, L, K, M, M), dtype=np.complex128)
    C = np.zeros((S, L, K, M, M), dtype=np.complex128)

    sqrt_pk_tau = np.sqrt(p_vec * tau_p).astype(float)  # (K,)
    pk_tau = (p_vec * tau_p).astype(float)              # (K,)

    for s in range(S):
        for k in range(K):
            t_k = int(pilot_of_user[s, k])
            if t_k < 0 or t_k >= tau_p:
                raise ValueError("pilot_of_user contains an out-of-range pilot index")

            Om_inv_l = Omega_inv[s, :, t_k]  # (L, M, M)

            for l in range(L):
                A = R_u[s, l, k] @ Om_inv_l[l]

                W[s, l, k] = sqrt_pk_tau[k] * A

                R_hat_lk = pk_tau[k] * (A @ R_u[s, l, k])
                R_hat_lk = 0.5 * (R_hat_lk + R_hat_lk.conj().T)

                C_lk = R_u[s, l, k] - R_hat_lk
                C_lk = 0.5 * (C_lk + C_lk.conj().T)

                R_hat[s, l, k] = R_hat_lk
                C[s, l, k] = C_lk

    return ChannelEstimationSetups(
        u_hat=None,
        R_u=R_u,
        Omega=Omega,
        Omega_inv=Omega_inv,
        W=W,
        R_hat=R_hat,
        C=C,
        pilot_of_user=pilot_of_user,
        chi=chi,
        tau_p=tau_p,
        pilot_power_watt_per_user=p_vec,
    )


def estimate_u_mmse_realizations(
    cfg: Config,
    channels: ChannelSetups,
    est_stats: ChannelEstimationSetups,
    pilot_groups: Optional[Sequence[Sequence[np.ndarray]]] = None,
    *,
    seed: Optional[int] = None,
    dtype_out: np.dtype = np.complex64,
    block_len: int = 256,
) -> ChannelEstimationSetups:
    """
    Compute realization-level MMSE estimates u_hat using already built estimation statistics.

    This is only needed for the Monte Carlo SINR branch.

    Inputs
    - channels.u_ap_user must exist with shape (S, L, M, K, R)
    - est_stats must already contain R_u, Omega, W, etc.
    """
    cfg.validate()

    if block_len <= 0:
        raise ValueError("block_len must be positive")

    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)
    R = int(cfg.sim.num_realizations)
    tau_p = int(est_stats.tau_p)

    u = np.asarray(channels.u_ap_user, dtype=dtype_out)
    if u.shape != (S, L, M, K, R):
        raise ValueError("channels.u_ap_user must have shape (S, L, M, K, R)")

    pilot_of_user = np.asarray(est_stats.pilot_of_user, dtype=int)
    if pilot_of_user.shape != (S, K):
        raise ValueError("est_stats.pilot_of_user must have shape (S, K)")

    if pilot_groups is None:
        pilot_groups_eff = _pilot_groups_from_indices(pilot_of_user, tau_p)
    else:
        if len(pilot_groups) != S:
            raise ValueError("pilot_groups must have length S")
        pilot_groups_eff = [list(g) for g in pilot_groups]

    W = np.asarray(est_stats.W, dtype=np.complex128)
    if W.shape != (S, L, K, M, M):
        raise ValueError("est_stats.W must have shape (S, L, K, M, M)")

    p_vec = np.asarray(est_stats.pilot_power_watt_per_user, dtype=float)
    if p_vec.shape != (K,):
        raise ValueError("est_stats.pilot_power_watt_per_user must have shape (K,)")

    sqrt_p_tau_all = np.sqrt(p_vec * tau_p).astype(float)  # (K,)

    u_hat = np.zeros((S, L, M, K, R), dtype=dtype_out)

    # Build estimates block by block over realizations
    for s in range(S):
        for l in range(L):
            for t in range(tau_p):
                users_t = np.asarray(pilot_groups_eff[s][t], dtype=int).ravel()

                for r0 in range(0, R, block_len):
                    r1 = min(R, r0 + block_len)
                    rb = r1 - r0

                    # y_lt = sum_{i in P_t} sqrt(p_i tau_p) u_li + n
                    if users_t.size == 0:
                        y = _complex_gaussian((M, rb), rng=rng, dtype=dtype_out)
                    else:
                        u_all_users = u[s, l, :, :, r0:r1]            # (M, K, rb)
                        u_slice = np.take(u_all_users, users_t, axis=1)  # (M, |P_t|, rb)
                        w_users = sqrt_p_tau_all[users_t]                # (|P_t|,)
                        y = np.tensordot(u_slice, w_users, axes=(1, 0))  # (M, rb)
                        y = y + _complex_gaussian((M, rb), rng=rng, dtype=dtype_out)

                    for k in users_t:
                        u_hat[s, l, :, k, r0:r1] = W[s, l, k].astype(dtype_out) @ y

    return ChannelEstimationSetups(
        u_hat=u_hat,
        R_u=est_stats.R_u,
        Omega=est_stats.Omega,
        Omega_inv=est_stats.Omega_inv,
        W=est_stats.W,
        R_hat=est_stats.R_hat,
        C=est_stats.C,
        pilot_of_user=est_stats.pilot_of_user,
        chi=est_stats.chi,
        tau_p=est_stats.tau_p,
        pilot_power_watt_per_user=est_stats.pilot_power_watt_per_user,
    )


def estimate_u_mmse_setups(
    cfg: Config,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    pilot_of_user: np.ndarray,
    pilot_groups: Optional[Sequence[Sequence[np.ndarray]]] = None,
    phases: Optional[PhaseShiftSetups] = None,
    *,
    channels: Optional[ChannelSetups] = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    dtype_out: np.dtype = np.complex64,
    block_len: int = 256,
) -> ChannelEstimationSetups:
    """
    Convenience wrapper.

    Behavior
    - always builds statistics
    - if channels is provided, also builds u_hat realizations
    - if channels is None, returns statistics-only object
    """
    est_stats = build_channel_estimation_statistics(
        cfg=cfg,
        corr=corr,
        ls=ls,
        pilot_of_user=pilot_of_user,
        pilot_groups=pilot_groups,
        phases=phases,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    if channels is None:
        return est_stats

    return estimate_u_mmse_realizations(
        cfg=cfg,
        channels=channels,
        est_stats=est_stats,
        pilot_groups=pilot_groups,
        seed=seed,
        dtype_out=dtype_out,
        block_len=block_len,
    )