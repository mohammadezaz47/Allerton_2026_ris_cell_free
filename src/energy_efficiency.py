# ITNG YMYFA 80asra YA
# IMZZ

# src/energy_efficiency.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import Config
from .power_allocation import PowerAllocationSetups
from .spectral_efficiency import SpectralEfficiencySetups



@dataclass(frozen=True)
class EnergyEfficiencySetups:
    """
    Energy efficiency outputs.

    Main outputs
    - ee_bit_per_joule: (S,)
    - ee_mbit_per_joule: (S,)
    - total_power_watt: (S,)
    - sum_rate_bps: (S,)

    Power breakdown
    - tx_power_watt: (S,)
    - circuit_power_watt: (S,)
    - ris_power_watt: (S,)
    - fronthaul_power_watt: (S,)

    Debug outputs
    - ap_active_mask: (S, L)
    - ap_tx_power_watt: (S, L)
    - user_serving_count: (S, K)
    - serve_mask_used: (S, L, K)
    """
    ee_bit_per_joule: np.ndarray
    ee_mbit_per_joule: np.ndarray

    total_power_watt: np.ndarray
    sum_rate_bps: np.ndarray

    tx_power_watt: np.ndarray
    circuit_power_watt: np.ndarray
    ris_power_watt: np.ndarray
    fronthaul_power_watt: np.ndarray

    ap_active_mask: np.ndarray
    ap_tx_power_watt: np.ndarray
    user_serving_count: np.ndarray
    serve_mask_used: np.ndarray

    tau_c: int
    tau_p: int
    tau_d: int
    bandwidth_hz: float
    pa_efficiency: float
    ap_circuit_power_watt_value: float
    ris_static_power_watt_value: float
    fronthaul_energy_per_bit_joule: float



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



def compute_energy_efficiency_setups(
      cfg: Config,
      se_out: SpectralEfficiencySetups,
      power_out: PowerAllocationSetups,
      *,
      serve_mask: Optional[np.ndarray] = None,
      serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,  
) -> EnergyEfficiencySetups:
    """
    Compute system energy efficiency in bit/Joule.

    Model
      EE = B * sum_k SE_k / P_T

    with
      P_T
      = sum_l delta_l * P_tx,l
        + P_c * sum_l delta_l
        + P_RIS
        + B * E_p * sum_k |L_k| * SE_k

    and
      P_tx,l
      = (tau_d / (tau_c * epsilon_PA)) * sum_{i in K_l} eta_li

    Notes
    - sum_rate_bps = B * sum_k SE_k
    - Watt = Joule/sec, so rate_bps / power_watt gives bit/Joule
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    se = np.asarray(se_out.se, dtype=float)
    if se.shape != (S, K):
        raise ValueError("se_out.se must have shape (S, K)")

    rate_bps = np.asarray(se_out.rate_bps, dtype=float)
    if rate_bps.shape != (S, K):
        raise ValueError("se_out.rate_bps must have shape (S, K)")

    eta = np.asarray(power_out.eta, dtype=float)
    if eta.shape != (S, L, K):
        raise ValueError("power_out.eta must have shape (S, L, K)")

    serve_mask_used = _validate_or_build_serve_mask(
        cfg=cfg,
        power=power_out,
        serve_mask=serve_mask,
        serve_users=serve_users,
    )  # (S, L, K) 

    tau_c = int(se_out.tau_c)
    tau_p = int(se_out.tau_p)
    tau_d = int(se_out.tau_d)
    bandwidth_hz = float(se_out.bandwidth_hz)

    if tau_c <= 0:
        raise ValueError("tau_c must be positive")
    if tau_d < 0:
        raise ValueError("tau_d must be non-negative")
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    
    epsilon_pa = float(cfg.ee.pa_efficiency)
    P_c = float(cfg.ee.ap_circuit_power_watt)
    P_ris_static = float(cfg.ee.ris_static_power_watt)
    E_p = float(cfg.ee.fronthaul_energy_per_bit_joule)

    # AP activity indicator delta_l
    ap_active_mask = np.any(serve_mask_used, axis=2)
    ap_active_float = ap_active_mask.astype(float)

    # Per-AP average transmit power
    # sum_{i in K_l} eta_li is implemented by masking eta with serve_mask
    eta_served = eta * serve_mask_used.astype(float)                    # (S, L, K)
    ap_eta_sum = np.sum(eta_served, axis=2)                             # (S, L)

    ap_tx_power_watt = (float(tau_d) / (float(tau_c) * epsilon_pa)) * ap_eta_sum  # (S, L)
    ap_tx_power_watt = ap_tx_power_watt * ap_active_float

    tx_power_watt = np.sum(ap_tx_power_watt, axis=1)               # (S,)
    circuit_power_watt = P_c * np.sum(ap_active_float, axis=1)     # (S,)

    # RIS static power
    if cfg.ee.use_ris_power_only_when_enabled:
        ris_scalar = P_ris_static if cfg.ris.enable_ris and cfg.dims.num_ris > 0 else 0.0
    else:
        ris_scalar = P_ris_static
    ris_power_watt = np.full(S, ris_scalar, dtype=float)

    # |L_k|, number of APs serving each user
    user_serving_count = np.sum(serve_mask_used, axis=1).astype(int)   # (S, K)

    # Fronthaul power
    # B * E_p * sum_k |L_k| * SE_k
    fronthaul_power_watt = bandwidth_hz * E_p * np.sum(user_serving_count * se, axis=1)

    # Total power and total rate
    total_power_watt = (
        tx_power_watt
        + circuit_power_watt
        + ris_power_watt
        + fronthaul_power_watt
    )  # (S,)

    sum_rate_bps = np.sum(rate_bps, axis=1)  # (S,)

    # EE in bit/Joule
    ee_bit_per_joule = np.zeros(S, dtype=float)
    positive_power = total_power_watt > 0
    ee_bit_per_joule[positive_power] = sum_rate_bps[positive_power] / total_power_watt[positive_power]

    # Readable scale
    ee_mbit_per_joule = ee_bit_per_joule / 1e6

    return EnergyEfficiencySetups(
        ee_bit_per_joule=ee_bit_per_joule,
        ee_mbit_per_joule=ee_mbit_per_joule,
        total_power_watt=total_power_watt,
        sum_rate_bps=sum_rate_bps,
        tx_power_watt=tx_power_watt,
        circuit_power_watt=circuit_power_watt,
        ris_power_watt=ris_power_watt,
        fronthaul_power_watt=fronthaul_power_watt,
        ap_active_mask=ap_active_mask,
        ap_tx_power_watt=ap_tx_power_watt,
        user_serving_count=user_serving_count,
        serve_mask_used=serve_mask_used,
        tau_c=tau_c,
        tau_p=tau_p,
        tau_d=tau_d,
        bandwidth_hz=bandwidth_hz,
        pa_efficiency=epsilon_pa,
        ap_circuit_power_watt_value=P_c,
        ris_static_power_watt_value=ris_scalar,
        fronthaul_energy_per_bit_joule=E_p,
    )
