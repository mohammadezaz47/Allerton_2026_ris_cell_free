# ITNG YMYFA 80asra YA
# IMZZ

# src/correlation_setups.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .geometry import GeometrySetups
from .large_scale import LargeScaleSetups
from .correlation_base import local_scattering_correlation, ris_correlation_matrix, trace_normalize


@dataclass(frozen=True)
class CorrelationSetups:
    """
    Correlation and covariance matrices for all setups.

    Shapes
    - S setups, L APs, K users, T RIS panels (0 or 1), M antennas per AP, N RIS elements
    """
    # Normalized correlation matrices
    delta_ap_user: np.ndarray   # (S, L, K, M, M)
    delta_ap_ris: np.ndarray    # (S, L, T, M, M)
    R_ris: np.ndarray           # (N, N)

    # Covariance matrices scaled by beta_over_noise
    R_ap_user: np.ndarray       # (S, L, K, M, M)
    R_ris_user: np.ndarray      # (S, T, K, N, N)
    Gamma_ap_ris: np.ndarray    # (S, L, T, N, N)

    # Angles (kept for compatibility, but not used with master Delta)
    theta_ap_user: np.ndarray   # (S, L, K)
    theta_ap_ris: np.ndarray    # (S, L, T)

    # The master Delta used everywhere
    delta_master: np.ndarray    # (M, M)


def build_correlation_setups(
    cfg: Config,
    geom: GeometrySetups,
    ls: LargeScaleSetups,
    seed: Optional[int] = None,
    master_theta_rad: float = 0.0,
) -> CorrelationSetups:
    """
    Build correlation and covariance matrices for all setups.

    Master-Delta model
    - We ignore angle dependence.
    - We compute a single AP-side correlation matrix Delta_master and reuse it everywhere.
    - tr(Delta_master) = M after trace normalization.

    Uses noise-normalized large-scale coefficients (beta_over_noise) from ls.
    """
    cfg.validate()
    _ = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    S, L, _ = geom.ap_xy.shape
    _, K, _ = geom.user_xy.shape
    T = geom.ris_xy.shape[1]
    M = int(cfg.dims.num_ap_antennas)

    N = int(cfg.dims.num_ris_elements) if T > 0 else 0

    # Build master Delta once
    Delta_master = local_scattering_correlation(
        M=M,
        theta=float(master_theta_rad),
        asd_deg=cfg.arrays.asd_azim_deg,
        antenna_spacing=cfg.arrays.ap_spacing_wavelength,
        distribution=cfg.arrays.ap_scattering_distribution,
        num_points=cfg.arrays.ap_scattering_integration_points,
    )
    delta_master = trace_normalize(Delta_master, target_trace=M).astype(np.complex128)

    # RIS base correlation matrix R_ris (normalized to trace N)
    if N > 0:
        R_ris = ris_correlation_matrix(
            num_h=int(cfg.dims.ris_n_hor),
            num_v=int(cfg.dims.ris_n_ver),
            d_h_wavelengths=cfg.arrays.ris_spacing_wavelength,
            d_v_wavelengths=cfg.arrays.ris_spacing_wavelength,
            wavelength_m=1.0,
        )
        R_ris = trace_normalize(R_ris, target_trace=N).astype(np.complex128)
    else:
        R_ris = np.empty((0, 0), dtype=np.complex128)

    # Fill delta_ap_user and delta_ap_ris with the same master Delta
    delta_ap_user = np.empty((S, L, K, M, M), dtype=np.complex128)
    delta_ap_user[...] = delta_master

    delta_ap_ris = np.empty((S, L, T, M, M), dtype=np.complex128)
    if T > 0:
        delta_ap_ris[...] = delta_master

    # Dummy angle arrays, kept for compatibility
    theta_ap_user = np.zeros((S, L, K), dtype=float)
    theta_ap_ris = np.zeros((S, L, T), dtype=float)

    # Covariances scaled by beta_over_noise
    R_ap_user = ls.ap_user_beta_over_noise[..., None, None] * delta_ap_user

    if T > 0:
        R_ris_user = ls.ris_user_beta_over_noise[..., None, None] * R_ris[None, None, None, :, :]
        Gamma_ap_ris = ls.ap_ris_beta_over_noise[..., None, None] * R_ris[None, None, None, :, :]
    else:
        R_ris_user = np.empty((S, 0, K, 0, 0), dtype=np.complex128)
        Gamma_ap_ris = np.empty((S, L, 0, 0, 0), dtype=np.complex128)

    return CorrelationSetups(
        delta_ap_user=delta_ap_user,
        delta_ap_ris=delta_ap_ris,
        R_ris=R_ris,
        R_ap_user=R_ap_user,
        R_ris_user=R_ris_user,
        Gamma_ap_ris=Gamma_ap_ris,
        theta_ap_user=theta_ap_user,
        theta_ap_ris=theta_ap_ris,
        delta_master=delta_master,
    )