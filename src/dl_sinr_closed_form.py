# ITNG YMYFA 80asra YA
# IMZZ

# src/dl_sinr_closed_form.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import Config
from .channel_estimation import ChannelEstimationSetups
from .correlation_setups import CorrelationSetups
from .large_scale import LargeScaleSetups
from .power_allocation import PowerAllocationSetups
from .ris_phase_shifts import PhaseShiftSetups


@dataclass(frozen=True)
class DownlinkSinrClosedFormSetups:
    """
    Closed-form downlink SINR outputs.

    Main outputs
    - sinr_dl: (S, K)
    - ds_power: (S, K)
    - uu_power: (S, K)
    - interference_power: (S, K)
    - ci_power: (S, K)
    - ni_power: (S, K)
    - noise_power: (S, K)

    Debug outputs
    - ds_complex: (S, K)
    - mu: (S, L, K)
    - a: (S, L, K)
    - b: (S, L, K)
    - c: (S, L, K, K) where c[s,l,kp,k] = c_{lk'|k}
    - chi: (S,)
    - iota: (S,)
    - serve_mask_used: (S, L, K)
    """
    sinr_dl: np.ndarray
    ds_power: np.ndarray
    uu_power: np.ndarray
    interference_power: np.ndarray
    ci_power: np.ndarray
    ni_power: np.ndarray
    noise_power: np.ndarray

    ds_complex: np.ndarray

    mu: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray

    chi: np.ndarray
    iota: np.ndarray

    serve_mask_used: np.ndarray


def _validate_or_build_serve_mask(
    cfg: Config,
    power: PowerAllocationSetups,
    serve_mask: Optional[np.ndarray],
    serve_users: Optional[Sequence[Sequence[Sequence[int]]]],
) -> np.ndarray:
    """
    Return a boolean serving mask with shape (S, L, K).
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

    if serve_users is not None:
        if len(serve_users) == L and S == 1:
            serve_users = [serve_users]  # type: ignore[assignment]

        if len(serve_users) != S:
            raise ValueError("serve_users must have length S")

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

    m = np.asarray(power.serve_mask_used)
    if m.shape != (S, L, K):
        raise ValueError("power.serve_mask_used must have shape (S, L, K)")
    return m.astype(bool, copy=False)


def _pilot_groups_from_indices(pilot_of_user: np.ndarray, tau_p: int) -> list[list[np.ndarray]]:
    """
    Build pilot groups from pilot indices.

    Returns
    - groups[s][t] = array of user indices using pilot t in setup s
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


def _compute_chi_iota_per_setup(
    cfg: Config,
    corr: CorrelationSetups,
    phases: Optional[PhaseShiftSetups],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute chi and iota for each setup.

      chi  = tr(Theta R Theta^H R)
      iota = tr((Theta R Theta^H R)^2)
    """
    S = int(cfg.sim.num_setups)

    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        return np.zeros(S, dtype=float), np.zeros(S, dtype=float)

    if phases is None:
        raise ValueError("phases must be provided when RIS is enabled")

    R = np.asarray(corr.R_ris, dtype=np.complex128)
    if R.size == 0:
        return np.zeros(S, dtype=float), np.zeros(S, dtype=float)

    vartheta = np.asarray(phases.vartheta, dtype=float)
    if vartheta.shape[0] != S:
        raise ValueError("phases.vartheta must have first dimension equal to num_setups")

    chi = np.zeros(S, dtype=float)
    iota = np.zeros(S, dtype=float)

    for s in range(S):
        d = np.exp(1j * vartheta[s])  # (N,)
        Theta_R_ThetaH = (d[:, None] * R) * np.conj(d[None, :])  # (N, N)
        X = Theta_R_ThetaH @ R
        chi[s] = _sanitize_power_scalar(f"chi[s={s}]", _trace_complex(X))
        iota[s] = _sanitize_power_scalar(f"iota[s={s}]", _trace_complex(X @ X))

    return chi, iota


def _get_delta_all(cfg: Config, corr: CorrelationSetups) -> np.ndarray:
    """
    Return Delta with shape (S, L, M, M).

    Policy
    - If RIS is active and corr.delta_ap_ris exists, use corr.delta_ap_ris[:, :, 0, :, :].
    - If RIS is disabled, keep the AP-side correlation nonzero by using corr.delta_master
      broadcast over setups and APs.

    This is the correct no-RIS behavior for your model, because Delta is the AP-side
    correlation structure and should not be forced to zero when the RIS is absent.
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    M = int(cfg.dims.num_ap_antennas)

    if cfg.ris.enable_ris and corr.delta_ap_ris.ndim >= 3 and corr.delta_ap_ris.shape[2] > 0:
        return np.asarray(corr.delta_ap_ris[:, :, 0, :, :], dtype=np.complex128)

    delta_master = np.asarray(corr.delta_master, dtype=np.complex128)
    if delta_master.shape != (M, M):
        raise ValueError("corr.delta_master must have shape (M, M)")

    delta_all = np.empty((S, L, M, M), dtype=np.complex128)
    delta_all[...] = delta_master
    return delta_all


def _trace_complex(A: np.ndarray) -> complex:
    """Convenience wrapper around matrix trace."""
    return complex(np.trace(A))


def _sanitize_real_scalar(
    name: str,
    value: complex,
    *,
    imag_tol: float = 1e-9,
) -> float:
    """
    Keep the real part of a scalar that is theoretically real.
    Warn only when the imaginary leakage is not negligible.
    """
    z = complex(value)
    imag_abs = float(abs(np.imag(z)))
    scale = max(1.0, float(abs(z)))

    if imag_abs > imag_tol * scale:
        print(
            f"[WARNING] {name} has non-negligible imaginary part: "
            f"real={np.real(z):.6e}, imag={np.imag(z):.6e}"
        )

    real_val = float(np.real(z))
    if not np.isfinite(real_val):
        raise RuntimeError(f"{name} became non-finite: {z}")

    return real_val


def _sanitize_power_scalar(
    name: str,
    value: complex,
    *,
    imag_tol: float = 1e-9,
    neg_tol: float = 1e-10,
) -> float:
    """
    Sanitize a scalar that is theoretically a power.
    Final result is real and non-negative.
    """
    z = complex(value)
    real_val = _sanitize_real_scalar(name, z, imag_tol=imag_tol)

    scale = max(1.0, float(abs(z)))
    if real_val < -neg_tol * scale:
        print(
            f"[WARNING] {name} has negative real part before clipping: "
            f"{real_val:.6e}"
        )

    return max(0.0, real_val)


def _build_a_b_mu_c(
    est: ChannelEstimationSetups,
    delta_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build helper coefficients

    - mu[s,l,k] = tr(R_hat[s,l,k])
    - a[s,l,k] = tr(W[s,l,k] Delta[s,l])
    - b[s,l,k] = tr(Delta[s,l] W[s,l,k] Delta[s,l] W[s,l,k]^H)
    - c[s,l,kp,k] = tr(W[s,l,kp]^H R_u[s,l,k] W[s,l,kp])
    """
    W = np.asarray(est.W, dtype=np.complex128)        # (S, L, K, M, M)
    R_u = np.asarray(est.R_u, dtype=np.complex128)    # (S, L, K, M, M)
    R_hat = np.asarray(est.R_hat, dtype=np.complex128)

    S, L, K, M, _ = W.shape

    mu = np.maximum(
        0.0,
        np.real(np.trace(R_hat, axis1=-2, axis2=-1)),
    )  # (S, L, K)

    a = np.zeros((S, L, K), dtype=np.complex128)
    b = np.zeros((S, L, K), dtype=np.complex128)
    c = np.zeros((S, L, K, K), dtype=np.complex128)

    for s in range(S):
        for l in range(L):
            Delta = delta_all[s, l]  # (M, M)

            for kp in range(K):
                Wlkp = W[s, l, kp]

                a[s, l, kp] = _trace_complex(Wlkp @ Delta)
                b[s, l, kp] = _trace_complex(Delta @ Wlkp @ Delta @ Wlkp.conj().T)

                Wlkp_H = Wlkp.conj().T
                for k in range(K):
                    c[s, l, kp, k] = _trace_complex(Wlkp_H @ R_u[s, l, k] @ Wlkp)

    return a, b, mu, c


def _zeta_lk_A(
    R_f_lk: np.ndarray,
    Delta: np.ndarray,
    A: np.ndarray,
    hat_beta_l: float,
    tilde_beta_k: float,
    chi: float,
    iota: float,
    Gamma_l: np.ndarray,
    Xi_k: np.ndarray,
) -> float:
    """
    Compute zeta_lk(A) using the definitions you provided.
    """
    A_H = A.conj().T

    # xi_l(Xi_k, A, A^H, Xi_k)
    tr_Xi_Gamma = _trace_complex(Xi_k @ Gamma_l)
    tr_A_Delta = _trace_complex(A @ Delta)
    tr_AH_Delta = _trace_complex(A_H @ Delta)
    tr_Xi_Gamma_Xi_Gamma = _trace_complex(Xi_k @ Gamma_l @ Xi_k @ Gamma_l)
    tr_A_Delta_AH_Delta = _trace_complex(A @ Delta @ A_H @ Delta)

    xi_val = (
        tr_Xi_Gamma * tr_A_Delta * tr_Xi_Gamma * tr_AH_Delta
        + tr_Xi_Gamma_Xi_Gamma * tr_A_Delta_AH_Delta
    )

    # alpha_lk(A)
    alpha_val = xi_val + (hat_beta_l ** 2) * (tilde_beta_k ** 2) * (
        tr_A_Delta * tr_AH_Delta * iota
        + _trace_complex(Delta @ A @ Delta @ A_H) * (chi ** 2)
    )

    # zeta_lk(A)
    zeta_val = (
        alpha_val
        + _trace_complex(R_f_lk @ A) * _trace_complex(R_f_lk @ A_H)
        + _trace_complex(R_f_lk @ A @ R_f_lk @ A_H)
        + (
            _trace_complex(A @ R_f_lk @ A_H @ Delta)
            + _trace_complex(A_H @ R_f_lk @ A @ Delta)
            + _trace_complex(R_f_lk @ A) * _trace_complex(A_H @ Delta)
            + _trace_complex(R_f_lk @ A_H) * _trace_complex(A @ Delta)
        ) * hat_beta_l * tilde_beta_k * chi
    )

    return _sanitize_power_scalar("zeta_lk_A", zeta_val)


def compute_dl_sinr_closed_form(
    cfg: Config,
    est: ChannelEstimationSetups,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    power: PowerAllocationSetups,
    pilot_of_user: np.ndarray,
    phases: Optional[PhaseShiftSetups],
    *,
    serve_mask: Optional[np.ndarray] = None,
    serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> DownlinkSinrClosedFormSetups:
    """
    Compute the closed-form downlink SINR using the theorem and lemmas you provided.

    Normalized-domain convention
    - sigma_ul^2 = 1
    - sigma_dl^2 = 1

    Single-RIS assumption
    - This implementation assumes at most one RIS is active.
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)
    tau_p = int(round(float(cfg.tau_p())))

    if tau_p <= 0:
        raise ValueError("tau_p must be positive")

    if pilot_of_user.shape != (S, K):
        raise ValueError("pilot_of_user must have shape (S, K)")

    T = 0 if corr.delta_ap_ris.ndim < 3 else int(corr.delta_ap_ris.shape[2])
    if cfg.ris.enable_ris and T > 1:
        raise NotImplementedError("Closed-form SINR currently supports only single-RIS setups")

    serve_mask_used = _validate_or_build_serve_mask(
        cfg=cfg,
        power=power,
        serve_mask=serve_mask,
        serve_users=serve_users,
    )  # (S, L, K)

    eta = np.asarray(power.eta, dtype=float)
    if eta.shape != (S, L, K):
        raise ValueError("power.eta must have shape (S, L, K)")

    # Pilot powers p_i
    if pilot_power_watt_per_user is None:
        p_vec = np.full(K, float(cfg.pilots.pilot_power_watt), dtype=float)
    else:
        p_vec = np.asarray(pilot_power_watt_per_user, dtype=float).reshape(-1)
        if p_vec.shape[0] != K:
            raise ValueError("pilot_power_watt_per_user must have length K")
        if np.any(p_vec < 0):
            raise ValueError("pilot powers must be non-negative")

    pilot_groups = _pilot_groups_from_indices(pilot_of_user, tau_p)

    # Setup-level RIS scalars
    chi, iota = _compute_chi_iota_per_setup(cfg, corr, phases)  # (S,), (S,)

    # Delta for all setups/APs
    delta_all = _get_delta_all(cfg, corr)  # (S, L, M, M)

    # Link-level helpers
    a, b, mu, c = _build_a_b_mu_c(est, delta_all)

    mu_safe = np.maximum(mu, eps)

    # Useful reused arrays
    R_u = np.asarray(est.R_u, dtype=np.complex128)
    R_f = np.asarray(corr.R_ap_user, dtype=np.complex128)

    beta_lk = np.asarray(ls.ap_user_beta_over_noise, dtype=float)  # (S, L, K)

    if cfg.ris.enable_ris and T > 0:
        hat_beta = np.asarray(ls.ap_ris_beta_over_noise[:, :, 0], dtype=float)   # (S, L)
        tilde_beta = np.asarray(ls.ris_user_beta_over_noise[:, 0, :], dtype=float)  # (S, K)
        R_ris = np.asarray(corr.R_ris, dtype=np.complex128)
    else:
        hat_beta = np.zeros((S, L), dtype=float)
        tilde_beta = np.zeros((S, K), dtype=float)
        R_ris = np.empty((0, 0), dtype=np.complex128)

    # Build Xi_k and Gamma_l when RIS is active
    Xi = None
    Gamma = None
    if cfg.ris.enable_ris and T > 0:
        Xi = np.zeros((S, K, R_ris.shape[0], R_ris.shape[1]), dtype=np.complex128)     # (S, K, N, N)
        Gamma = np.zeros((S, L, R_ris.shape[0], R_ris.shape[1]), dtype=np.complex128)   # (S, L, N, N)

        vartheta = np.asarray(phases.vartheta, dtype=float)
        for s in range(S):
            d = np.exp(1j * vartheta[s])  # (N,)
            Theta_R_ThetaH = (d[:, None] * R_ris) * np.conj(d[None, :])  # (N, N)

            for k in range(K):
                Xi[s, k] = tilde_beta[s, k] * Theta_R_ThetaH

            for l in range(L):
                Gamma[s, l] = hat_beta[s, l] * R_ris

    # Outputs
    sinr_dl = np.zeros((S, K), dtype=float)
    ds_power = np.zeros((S, K), dtype=float)
    uu_power = np.zeros((S, K), dtype=float)
    interference_power = np.zeros((S, K), dtype=float)
    ci_power = np.zeros((S, K), dtype=float)
    ni_power = np.zeros((S, K), dtype=float)
    noise_power = np.ones((S, K), dtype=float)  # sigma_dl^2 = 1

    ds_complex = np.zeros((S, K), dtype=np.complex128)

    # Main closed-form computation
    for s in range(S):
        for k in range(K):
            serving_k = np.where(serve_mask_used[s, :, k])[0].astype(int)
            pilot_k = int(pilot_of_user[s, k])
            users_same_pilot_k = np.asarray(pilot_groups[s][pilot_k], dtype=int)

            # Desired signal
            ds_amp = (
                np.sum(np.sqrt(eta[s, serving_k, k] * mu_safe[s, serving_k, k]))
                if serving_k.size > 0 else 0.0
            )
            ds_complex[s, k] = complex(ds_amp, 0.0)
            ds_power[s, k] = float(np.abs(ds_complex[s, k]) ** 2)

            # ---------------- UU term ----------------
            uu_k = 0.0 + 0.0j

            # Cross-AP part
            for idx_l, l in enumerate(serving_k):
                for idx_m, m in enumerate(serving_k):
                    if m == l:
                        continue

                    pref = np.sqrt(
                        (eta[s, l, k] * eta[s, m, k]) /
                        (mu_safe[s, l, k] * mu_safe[s, m, k])
                    )

                    term1 = (
                        p_vec[k] * tau_p
                        * (
                            beta_lk[s, l, k] * a[s, l, k]
                            + hat_beta[s, l] * tilde_beta[s, k] * a[s, l, k] * chi[s]
                        )
                        * np.conj(
                            beta_lk[s, m, k] * a[s, m, k]
                            + hat_beta[s, m] * tilde_beta[s, k] * a[s, m, k] * chi[s]
                        )
                    )

                    same_pilot_sum = 0.0
                    for i in users_same_pilot_k:
                        same_pilot_sum += (
                            p_vec[i] * tau_p
                            * hat_beta[s, l] * hat_beta[s, m]
                            * tilde_beta[s, k] * tilde_beta[s, i]
                            * a[s, l, k] * np.conj(a[s, m, k]) * iota[s]
                        )

                    uu_k += pref * (term1 + same_pilot_sum - mu[s, l, k] * mu[s, m, k])

            # Diagonal-AP part
            for l in serving_k:
                pref = eta[s, l, k] / mu_safe[s, l, k]

                if cfg.ris.enable_ris and T > 0:
                    zeta_val = _zeta_lk_A(
                        R_f_lk=R_f[s, l, k],
                        Delta=delta_all[s, l],
                        A=est.W[s, l, k],
                        hat_beta_l=hat_beta[s, l],
                        tilde_beta_k=tilde_beta[s, k],
                        chi=chi[s],
                        iota=iota[s],
                        Gamma_l=Gamma[s, l],
                        Xi_k=Xi[s, k],
                    )
                else:
                    # No-RIS case: Xi=0 and Gamma=0, but Delta remains the AP-side master correlation.
                    A = est.W[s, l, k]
                    A_H = A.conj().T
                    zeta_raw = (
                        _trace_complex(R_f[s, l, k] @ A) * _trace_complex(R_f[s, l, k] @ A_H)
                        + _trace_complex(R_f[s, l, k] @ A @ R_f[s, l, k] @ A_H)
                    )
                    zeta_val = _sanitize_power_scalar(
                        f"zeta_no_ris[s={s},l={l},k={k}]",
                        zeta_raw,
                    )

                diag_term = (
                    p_vec[k] * tau_p * zeta_val
                    + c[s, l, k, k] * 1.0
                    - (mu[s, l, k] ** 2)
                )

                extra_sum = 0.0 + 0.0j
                for i in users_same_pilot_k:
                    if i == k:
                        continue

                    extra_sum += (
                        p_vec[i] * tau_p
                        * (
                            beta_lk[s, l, k] * hat_beta[s, l] * tilde_beta[s, i] * b[s, l, k] * chi[s]
                            + beta_lk[s, l, k] * beta_lk[s, l, i] * b[s, l, k]
                            + (hat_beta[s, l] ** 2) * tilde_beta[s, k] * tilde_beta[s, i]
                            * (np.abs(a[s, l, k]) ** 2 * iota[s] + b[s, l, k] * (chi[s] ** 2))
                        )
                    )

                uu_k += pref * (diag_term + extra_sum)

            uu_power[s, k] = _sanitize_power_scalar(
                f"uu_power[s={s},k={k}]",
                uu_k,
            )

            # ---------------- Interference terms ----------------
            ci_k = 0.0 + 0.0j
            ni_k = 0.0 + 0.0j

            for kp in range(K):
                if kp == k:
                    continue

                serving_kp = np.where(serve_mask_used[s, :, kp])[0].astype(int)
                if serving_kp.size == 0:
                    continue

                pilot_kp = int(pilot_of_user[s, kp])
                users_same_pilot_kp = np.asarray(pilot_groups[s][pilot_kp], dtype=int)

                inter_kp = 0.0 + 0.0j

                # Cross-AP part
                for l in serving_kp:
                    for m in serving_kp:
                        if m == l:
                            continue

                        pref = np.sqrt(
                            (eta[s, l, kp] * eta[s, m, kp]) /
                            (mu_safe[s, l, kp] * mu_safe[s, m, kp])
                        )

                        if pilot_kp == pilot_k:
                            term1 = (
                                p_vec[k] * tau_p
                                * (
                                    beta_lk[s, l, k] * a[s, l, kp]
                                    + hat_beta[s, l] * tilde_beta[s, k] * a[s, l, kp] * chi[s]
                                )
                                * np.conj(
                                    beta_lk[s, m, k] * a[s, m, kp]
                                    + hat_beta[s, m] * tilde_beta[s, k] * a[s, m, kp] * chi[s]
                                )
                            )
                        else:
                            term1 = 0.0

                        same_pilot_sum = 0.0
                        for i in users_same_pilot_kp:
                            same_pilot_sum += (
                                p_vec[i] * tau_p
                                * hat_beta[s, l] * hat_beta[s, m]
                                * tilde_beta[s, k] * tilde_beta[s, i]
                                * a[s, l, kp] * np.conj(a[s, m, kp]) * iota[s]
                            )

                        inter_kp += pref * (term1 + same_pilot_sum)

                # Diagonal-AP part
                for l in serving_kp:
                    pref = eta[s, l, kp] / mu_safe[s, l, kp]

                    if cfg.ris.enable_ris and T > 0:
                        zeta_val = _zeta_lk_A(
                            R_f_lk=R_f[s, l, k],
                            Delta=delta_all[s, l],
                            A=est.W[s, l, kp],
                            hat_beta_l=hat_beta[s, l],
                            tilde_beta_k=tilde_beta[s, k],
                            chi=chi[s],
                            iota=iota[s],
                            Gamma_l=Gamma[s, l],
                            Xi_k=Xi[s, k],
                        )
                    else:
                        A = est.W[s, l, kp]
                        A_H = A.conj().T
                        zeta_raw = (
                            _trace_complex(R_f[s, l, k] @ A) * _trace_complex(R_f[s, l, k] @ A_H)
                            + _trace_complex(R_f[s, l, k] @ A @ R_f[s, l, k] @ A_H)
                        )
                        zeta_val = _sanitize_power_scalar(
                            f"zeta_no_ris[s={s},l={l},k={k},kp={kp}]",
                            zeta_raw,
                        )

                    diag_term = c[s, l, kp, k] * 1.0

                    if pilot_kp == pilot_k:
                        diag_term += p_vec[k] * tau_p * zeta_val

                    extra_sum = 0.0 + 0.0j
                    for i in users_same_pilot_kp:
                        if pilot_kp == pilot_k and i == k:
                            continue

                        extra_sum += (
                            p_vec[i] * tau_p
                            * (
                                beta_lk[s, l, k] * hat_beta[s, l] * tilde_beta[s, i] * b[s, l, kp] * chi[s]
                                + beta_lk[s, l, i] * hat_beta[s, l] * tilde_beta[s, k] * b[s, l, kp] * chi[s]
                                + beta_lk[s, l, k] * beta_lk[s, l, i] * b[s, l, kp]
                                + (hat_beta[s, l] ** 2) * tilde_beta[s, k] * tilde_beta[s, i]
                                * (np.abs(a[s, l, kp]) ** 2 * iota[s] + b[s, l, kp] * (chi[s] ** 2))
                            )
                        )

                    inter_kp += pref * (diag_term + extra_sum)

                if pilot_kp == pilot_k:
                    ci_k += inter_kp
                else:
                    ni_k += inter_kp

            ci_power[s, k] = _sanitize_power_scalar(
                f"ci_power[s={s},k={k}]",
                ci_k,
            )

            ni_power[s, k] = _sanitize_power_scalar(
                f"ni_power[s={s},k={k}]",
                ni_k,
            )
            interference_power[s, k] = max(0.0, ci_power[s, k] + ni_power[s, k])

            denom = uu_power[s, k] + interference_power[s, k] + noise_power[s, k]
            sinr_dl[s, k] = ds_power[s, k] / max(denom, eps)

    return DownlinkSinrClosedFormSetups(
        sinr_dl=sinr_dl,
        ds_power=ds_power,
        uu_power=uu_power,
        interference_power=interference_power,
        ci_power=ci_power,
        ni_power=ni_power,
        noise_power=noise_power,
        ds_complex=ds_complex,
        mu=mu,
        a=a,
        b=b,
        c=c,
        chi=chi,
        iota=iota,
        serve_mask_used=serve_mask_used,
    )