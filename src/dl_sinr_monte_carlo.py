# ITNG YMYFA 80asra YA
# IMZZ

# src/dl_sinr_monte_carlo

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import Config
from .channel_estimation import ChannelEstimationSetups
from .channels import ChannelSetups
from .power_allocation import PowerAllocationSetups


@dataclass(frozen=True)
class DownlinkSinrMonteCarloSetups:
    """
    Monte Carlo downlink SINR outputs.

    Dimensions
    - S: setups
    - L: APs
    - K: users
    - M: antennas per AP
    - R: realizations

    Main outputs
    - sinr_dl: (S, K)
    - ds_power: (S, K)
    - uu_power: (S, K)
    - interference_power: (S, K)
    - ci_power: (S, K)
    - ni_power: (S, K)
    - noise_power: (S, K)

    Useful debug outputs
    - ds_complex: (S, K)
    - g_bar: (S, L, K)
    - mu: (S, L, K)
    - avg_precoder_norm: (S, L, K)

    The serving mask used internally is also returned for clarity.
    """
    sinr_dl: np.ndarray
    ds_power: np.ndarray
    uu_power: np.ndarray
    interference_power: np.ndarray
    ci_power: np.ndarray
    ni_power: np.ndarray
    noise_power: np.ndarray

    ds_complex: np.ndarray
    g_bar: np.ndarray
    mu: np.ndarray
    avg_precoder_norm: np.ndarray

    serve_mask_used: np.ndarray



def _validate_or_build_serve_mask(
    cfg: Config,
    power: PowerAllocationSetups,
    serve_mask: Optional[np.ndarray],
    serve_users: Optional[Sequence[Sequence[Sequence[int]]]],
) -> np.ndarray:
    """
    Return a boolean serving mask with shape (S, L, K).

    Priority
    1. serve_mask if provided
    2. serve_users if provided
    3. power.serve_mask_used otherwise
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



def compute_dl_sinr_monte_carlo(
        cfg: Config,
        channels: ChannelSetups,
        est: ChannelEstimationSetups,
        power: PowerAllocationSetups,
        pilot_of_user: np.ndarray,
        *,
        serve_mask: Optional[np.ndarray] = None,
        serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        eps: float = 1e-12,
) -> DownlinkSinrMonteCarloSetups:
    """
    Compute Monte Carlo downlink SINR and its components.

    Normalized-domain convention
    - sigma_dl^2 = 1

    Precoder
      w_lk(r) = u_hat_lk(r) / sqrt(mu_lk)
    where
      mu_lk = tr(R_hat_lk)

    Desired signal term
      DS_k = sum_{l in L_k} sqrt(eta_lk) E[u_lk^H w_lk]

    User uncertainty
      UU_k(r) = sum_{l in L_k} sqrt(eta_lk) (u_lk(r)^H w_lk(r) - E[u_lk^H w_lk])

    Interference from user k'
      I_kk'(r) = sum_{l in L_k'} sqrt(eta_lk') u_lk(r)^H w_lk'(r)

    Coherent interference
    - k' shares the same pilot as k

    Non-coherent interference
    - k' uses a different pilot than k
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)
    R = int(cfg.sim.num_realizations)

    # Inputs
    u = np.asarray(channels.u_ap_user)
    if u.shape != (S, L, M, K, R):
        raise ValueError("channels.u_ap_user must have shape (S, L, M, K, R)")

    if est.u_hat is None:
        raise ValueError(
            "Monte Carlo SINR requires realization-level estimates. "
            "Call estimate_u_mmse_realizations(...) or estimate_u_mmse_setups(..., channels=channels)."
        )

    u_hat = np.asarray(est.u_hat)
    if u_hat.shape != (S, L, M, K, R):
        raise ValueError("est.u_hat must have shape (S, L, M, K, R)")

    R_hat = np.asarray(est.R_hat)
    if R_hat.shape != (S, L, K, M, M):
        raise ValueError("est.R_hat must have shape (S, L, K, M, M)")

    eta = np.asarray(power.eta, dtype=float)
    if eta.shape != (S, L, K):
        raise ValueError("power.eta must have shape (S, L, K)")

    pilot_of_user = np.asarray(pilot_of_user, dtype=int)
    if pilot_of_user.shape != (S, K):
        raise ValueError("pilot_of_user must have shape (S, K)")

    serve_mask_used = _validate_or_build_serve_mask(
        cfg=cfg,
        power=power,
        serve_mask=serve_mask,
        serve_users=serve_users,
    )  # (S, L, K)

    # mu_lk = tr(R_hat_lk)
    mu = np.real(np.trace(R_hat, axis1=-2, axis2=-1))       # (S, L, K)
    mu_safe = np.maximum(mu, eps)

    # w_lk(r) = u_hat_lk(r) / sqrt(mu_lk)
    sqrt_mu = np.sqrt(mu_safe)[:, :, None, :, None]         # (S, L, 1, K, 1)
    w = u_hat / sqrt_mu                                     # (S, L, M, K, R)

    # If mu is numerically zero, force w to zero
    zero_mu_mask = (mu <= eps)[:, :, None, :, None]
    w = np.where(zero_mu_mask, 0.0, w)

    # Debug sanity
    avg_precoder_norm = np.mean(np.sum(np.abs(w) ** 2, axis=2), axis=-1)  # (S, L, K)

     # Outputs
    sinr_dl = np.zeros((S, K), dtype=float)
    ds_power = np.zeros((S, K), dtype=float)
    uu_power = np.zeros((S, K), dtype=float)
    interference_power = np.zeros((S, K), dtype=float)
    ci_power = np.zeros((S, K), dtype=float)
    ni_power = np.zeros((S, K), dtype=float)
    noise_power = np.ones((S, K), dtype=float)  # sigma_dl^2 = 1 in normalized domain

    ds_complex = np.zeros((S, K), dtype=np.complex128)
    g_bar = np.zeros((S, L, K), dtype=np.complex128)


    # Main Monte Carlo loops
    for s in range(S):
        for k in range(K):
            # APs serving user k
            serving_k = serve_mask_used[s, :, k]            # (L,)
            sqrt_eta_k = np.sqrt(eta[s, :, k])              # (L,)

            # Own effective beam inner products
            # scalars_self[l, r] = u_lk(r)^H w_lk(r)
            scalars_self = np.sum(np.conj(u[s, :, :, k, :]) * w[s, :, :, k, :], axis=1)  # (L, R)

            # Mean value per AP
            g_bar[s, :, k] = np.mean(scalars_self, axis=1)  # (L,)

            # Desired signal
            ds_k = np.sum(sqrt_eta_k[serving_k] * g_bar[s, serving_k, k])
            ds_complex[s, k] = ds_k
            ds_power[s, k] = np.abs(ds_k) ** 2

            # User uncertainty
            uu_realizations = np.sum(
                sqrt_eta_k[serving_k, None]
                * (scalars_self[serving_k, :] - g_bar[s, serving_k, k][:, None]),
                axis=0,
            )  # (R,)
            uu_power[s, k] = float(np.mean(np.abs(uu_realizations) ** 2))

            # Interference from every other user
            pilot_k = int(pilot_of_user[s, k])

            total_i = 0.0
            total_ci = 0.0
            total_ni = 0.0

            # u_lk(r)^H w_lk'(r) for all k' is needed user by user
            u_target = u[s, :, :, k, :]  # (L, M, R)

            for kp in range(K):
                if kp == k:
                    continue

                serving_kp = serve_mask_used[s, :, kp]      # (L,)
                if not np.any(serving_kp):
                    continue

                sqrt_eta_kp = np.sqrt(eta[s, :, kp])        # (L,)

                # scalars_int[l, r] = u_lk(r)^H w_lk'(r)
                scalars_int = np.sum(np.conj(u_target) * w[s, :, :, kp, :], axis=1)  # (L, R)

                i_realizations = np.sum(
                    sqrt_eta_kp[serving_kp, None] * scalars_int[serving_kp, :],
                    axis=0,
                )  # (R,)

                i_power = float(np.mean(np.abs(i_realizations) ** 2))
                total_i += i_power

                if int(pilot_of_user[s, kp]) == pilot_k:
                    total_ci += i_power
                else:
                    total_ni += i_power

            interference_power[s, k] = total_i
            ci_power[s, k] = total_ci
            ni_power[s, k] = total_ni

            denom = uu_power[s, k] + interference_power[s, k] + noise_power[s, k]
            sinr_dl[s, k] = ds_power[s, k] / max(denom, eps)

    return DownlinkSinrMonteCarloSetups(
        sinr_dl=sinr_dl,
        ds_power=ds_power,
        uu_power=uu_power,
        interference_power=interference_power,
        ci_power=ci_power,
        ni_power=ni_power,
        noise_power=noise_power,
        ds_complex=ds_complex,
        g_bar=g_bar,
        mu=mu,
        avg_precoder_norm=avg_precoder_norm,
        serve_mask_used=serve_mask_used,
    )