# ITNG YMYFA 80asra YA
# IMZZ

# src/large_scale.py

"""
large_scale.py

Compute large-scale fading for all links, using cached distances from geometry.

Outputs
- gain_db: large-scale gain in dB (bigger is stronger)
- beta: linear scale gain, beta = 10^(gain_db/10)
- beta_over_noise: beta normalized by noise power in Watt

Link types and shapes
- AP–User: (S, L, K)
- AP–RIS:  (S, L, T)
- RIS–User: (S, T, K)

T is 0 when RIS is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .config import Config, dbm_to_watt
from .geometry import GeometrySetups



@dataclass(frozen=True)
class LargeScaleSetups:
    """Large-scale outputs for all setups."""
    # Gains in dB
    ap_user_gain_db: np.ndarray   # (S, L, K)
    ap_ris_gain_db: np.ndarray    # (S, L, T)
    ris_user_gain_db: np.ndarray  # (S, T, K)

    # Linear betas
    ap_user_beta: np.ndarray      # (S, L, K)
    ap_ris_beta: np.ndarray       # (S, L, T)
    ris_user_beta: np.ndarray     # (S, T, K)

    # Noise-normalized (linear)
    ap_user_beta_over_noise: np.ndarray   # (S, L, K)
    ap_ris_beta_over_noise: np.ndarray    # (S, L, T)
    ris_user_beta_over_noise: np.ndarray  # (S, T, K)

    # Link state
    is_los_ap_user: np.ndarray    # (S, L, K)
    is_los_ap_ris: np.ndarray     # (S, L, T)
    is_los_ris_user: np.ndarray   # (S, T, K)

    # Optional blockage info for AP–User
    is_blocked_ap_user: np.ndarray  # (S, L, K)

    # Noise for convenience
    noise_power_dbm: float
    noise_power_watt: float



def _decide_los_mask(
        rng: np.random.Generator,
        mode: Literal["los", "nlos", "probabilistic"],
        d_m: np.ndarray,
        max_los_distance_m: float,
) -> np.ndarray:
    """
    Decide LoS vs NLoS for every entry in d_m.

    probabilistic rule
    p_los = clip(1 - d/max_los_distance, 0, 1)
    """
    if mode == "los":
        return np.ones_like(d_m, dtype=bool)
    
    if mode == "nlos":
        return np.zeros_like(d_m, dtype=bool)
    
    if max_los_distance_m <= 0:
        raise ValueError("max_los_distance_m must be positive for probabilistic mode")

    p_los = 1.0 - (d_m / float(max_los_distance_m))
    p_los = np.clip(p_los, 0.0, 1.0)

    return rng.random(size=d_m.shape) < p_los



def _gain_db_from_distance(
        rng: np.random.Generator,
        d_m: np.ndarray,
        is_los: np.ndarray,
        cfg: Config,
) -> np.ndarray:
    """
    Compute gain in dB given distance and LoS mask.

    LoS
      gain_db = pl_los_intercept_db + pl_los_slope_db_per_log10m * log10(d) + sigma_los * N(0,1)

    NLoS
      gain_db = pl_nlos_intercept_db + pl_nlos_slope_db_per_log10m * log10(d) + sigma_nlos * N(0,1)
    """
    d_m = np.asarray(d_m, dtype=float)
    is_los = np.asarray(is_los, dtype=bool)

    z_los = rng.standard_normal(size=d_m.shape)
    z_nlos = rng.standard_normal(size=d_m.shape)

    gain_los = (
        cfg.prop.pl_los_intercept_db
        + cfg.prop.pl_los_slope_db_per_log10m * np.log10(d_m)
        + cfg.prop.shadow_std_los_db * z_los
    )

    gain_nlos = (
        cfg.prop.pl_nlos_intercept_db
        + cfg.prop.pl_nlos_slope_db_per_log10m * np.log10(d_m)
        + cfg.prop.shadow_std_nlos_db * z_nlos
    )

    return np.where(is_los, gain_los, gain_nlos)



def _db_to_linear(x_db: np.ndarray) -> np.ndarray:
    """10^(x_db/10)"""
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)



def compute_large_scale_setups(
    cfg: Config,
    geom: GeometrySetups,
    seed: Optional[int] = None,
) -> LargeScaleSetups:
    """
    Compute large-scale fading coefficients for all setups using cached distances from geom.

    Uses
    - geom.dist.ap_user_3d with shape (S, L, K)
    - geom.dist.ap_ris_3d with shape (S, L, T)
    - geom.dist.ris_user_3d with shape (S, T, K)
    """
    cfg.validate()

    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    # Noise power
    noise_dbm = float(cfg.noise.noise_power_dbm)
    noise_watt = float(dbm_to_watt(noise_dbm))

    # Distances (already include wrap-around effects through geometry)
    d_ap_user = np.maximum(geom.dist.ap_user_3d, float(cfg.prop.min_distance_m))    # (S, L, K)

    # RIS distances may have T = 0
    d_ap_ris = geom.dist.ap_ris_3d
    d_ris_user = geom.dist.ris_user_3d

    if d_ap_ris.size > 0:
        d_ap_ris = np.maximum(d_ap_ris, float(cfg.prop.min_distance_m))  # (S, L, T)
    if d_ris_user.size > 0:
        d_ris_user = np.maximum(d_ris_user, float(cfg.prop.min_distance_m))  # (S, T, K)

    # LoS masks
    is_los_ap_user = _decide_los_mask(
        rng=rng,
        mode=cfg.prop.ap_user_condition,
        d_m=d_ap_user,
        max_los_distance_m=cfg.prop.max_los_distance_m,
    )

    if d_ap_ris.size > 0:
        is_los_ap_ris = _decide_los_mask(
            rng=rng,
            mode=cfg.prop.ap_ris_condition,
            d_m=d_ap_ris,
            max_los_distance_m=cfg.prop.max_los_distance_m,
        )
    else:
        is_los_ap_ris = np.empty_like(d_ap_ris, dtype=bool)

    if d_ris_user.size > 0:
        is_los_ris_user = _decide_los_mask(
            rng=rng,
            mode=cfg.prop.ris_user_condition,
            d_m=d_ris_user,
            max_los_distance_m=cfg.prop.max_los_distance_m,
        )
    else:
        is_los_ris_user = np.empty_like(d_ris_user, dtype=bool)

     # Gains in dB
    ap_user_gain_db = _gain_db_from_distance(rng, d_ap_user, is_los_ap_user, cfg)

    if d_ap_ris.size > 0:
        ap_ris_gain_db = _gain_db_from_distance(rng, d_ap_ris, is_los_ap_ris, cfg)
    else:
        ap_ris_gain_db = np.empty_like(d_ap_ris, dtype=float)

    if d_ris_user.size > 0:
        ris_user_gain_db = _gain_db_from_distance(rng, d_ris_user, is_los_ris_user, cfg)
    else:
        ris_user_gain_db = np.empty_like(d_ris_user, dtype=float)

    # Always-applied extra loss on the direct link if configured (default is 0.0)
    if float(cfg.prop.direct_loss_db) != 0.0:
        ap_user_gain_db = ap_user_gain_db - float(cfg.prop.direct_loss_db)
    
    # Direct link blockage on AP–User if configured
    p_block = float(cfg.prop.ap_user_block_prob)
    if p_block > 0.0:
        is_blocked_ap_user = rng.random(size=ap_user_gain_db.shape) < p_block

        if cfg.prop.blockage_mode == "extra_loss_db":
            ap_user_gain_db = np.where(
                is_blocked_ap_user,
                ap_user_gain_db - float(cfg.prop.blockage_extra_loss_db),
                ap_user_gain_db,
            )
        elif cfg.prop.blockage_mode == "zero":
            ap_user_gain_db = np.where(is_blocked_ap_user, -np.inf, ap_user_gain_db)
        else:
            raise ValueError(f"Unknown blockage_mode {cfg.prop.blockage_mode}")
    else:
        is_blocked_ap_user = np.zeros_like(ap_user_gain_db, dtype=bool)

    # Linear betas
    ap_user_beta = _db_to_linear(ap_user_gain_db)
    ap_ris_beta = _db_to_linear(ap_ris_gain_db) if ap_ris_gain_db.size > 0 else ap_ris_gain_db
    ris_user_beta = _db_to_linear(ris_user_gain_db) if ris_user_gain_db.size > 0 else ris_user_gain_db

    # Noise-normalized
    ap_user_beta_over_noise = ap_user_beta / noise_watt
    ap_ris_beta_over_noise = (ap_ris_beta / noise_watt) if ap_ris_beta.size > 0 else ap_ris_beta
    ris_user_beta_over_noise = (ris_user_beta / noise_watt) if ris_user_beta.size > 0 else ris_user_beta

    return LargeScaleSetups(
        ap_user_gain_db=ap_user_gain_db,
        ap_ris_gain_db=ap_ris_gain_db,
        ris_user_gain_db=ris_user_gain_db,
        ap_user_beta=ap_user_beta,
        ap_ris_beta=ap_ris_beta,
        ris_user_beta=ris_user_beta,
        ap_user_beta_over_noise=ap_user_beta_over_noise,
        ap_ris_beta_over_noise=ap_ris_beta_over_noise,
        ris_user_beta_over_noise=ris_user_beta_over_noise,
        is_los_ap_user=is_los_ap_user,
        is_los_ap_ris=is_los_ap_ris,
        is_los_ris_user=is_los_ris_user,
        is_blocked_ap_user=is_blocked_ap_user,
        noise_power_dbm=noise_dbm,
        noise_power_watt=noise_watt,
    )