# ITNG YMYFA 80asra YA
# IMZZ

# src/spectral_efficiency.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config



@dataclass(frozen=True)
class SpectralEfficiencySetups:
    """
    Spectral efficiency and rate outputs.

    Shapes
    - sinr: (S, K)
    - se: (S, K)
    - rate_bps: (S, K)

    Setup-level summaries
    - sum_se: (S,)
    - sum_rate_bps: (S,)
    - avg_se_per_user: (S,)
    - avg_rate_per_user_bps: (S,)
    """
    sinr: np.ndarray
    se: np.ndarray
    rate_bps: np.ndarray

    sum_se: np.ndarray
    sum_rate_bps: np.ndarray
    avg_se_per_user: np.ndarray
    avg_rate_per_user_bps: np.ndarray

    prelog_factor: float
    tau_c: float
    tau_p: float
    tau_d: float
    bandwidth_hz: float



def compute_spectral_efficiency_setups(
        cfg: Config,
        sinr: np.ndarray,
        tau_d: Optional[float] = None,
) -> SpectralEfficiencySetups:
    """
    Compute downlink spectral efficiency and rate from SINR.

    Formula
      SE_k = (tau_d / tau_c) * log2(1 + gamma_k)

    Default
    - If tau_d is not provided, we use tau_d = tau_c - tau_p

    Inputs
    - sinr must have shape (S, K)

    Outputs
    - se in bits/sec/Hz
    - rate_bps in bits/sec
    """
    cfg.validate()

    sinr = np.asarray(sinr, dtype=float)

    S = int(cfg.sim.num_setups)
    K = int(cfg.dims.num_users)

    if sinr.shape != (S, K):
        raise ValueError("sinr must have shape (S, K)")
    
    tau_c = float(cfg.pilots.coherence_block_length)
    tau_p = float(cfg.tau_p())

    if tau_c <= 0:
        raise ValueError("coherence block length tau_c must be positive")
    if tau_p < 0:
        raise ValueError("pilot length tau_p must be non-negative")
    if tau_p > tau_c:
        raise ValueError("tau_p cannot be larger than tau_c")

    if tau_d is None:
        tau_d_eff = tau_c - tau_p
    else:
        tau_d_eff = float(tau_d)

    if tau_d_eff < 0:
        raise ValueError("tau_d must be non-negative")
    
    prelog_factor = float(tau_d_eff) / float(tau_c)
    bandwidth_hz = float(cfg.noise.bandwidth_hz)

    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")

    # Small negative values can appear from numerical noise in other modules
    sinr_safe = np.maximum(sinr, 0.0)

    se = prelog_factor * np.log2(1.0 + sinr_safe)       # (S, K)
    rate_bps = bandwidth_hz * se                        # (S, K)

    sum_se = np.sum(se, axis=1)                         # (S,)
    sum_rate_bps = np.sum(rate_bps, axis=1)             # (S,)
    avg_se_per_user = np.mean(se, axis=1)               # (S,)
    avg_rate_per_user_bps = np.mean(rate_bps, axis=1)   # (S,)

    return SpectralEfficiencySetups(
        sinr = sinr_safe,
        se=se,
        rate_bps=rate_bps,
        sum_se=sum_se,
        sum_rate_bps=sum_rate_bps,
        avg_se_per_user=avg_se_per_user,
        avg_rate_per_user_bps=avg_rate_per_user_bps,
        prelog_factor=prelog_factor,
        tau_c=tau_c,
        tau_p=tau_p,
        tau_d=tau_d_eff,
        bandwidth_hz=bandwidth_hz,
    )