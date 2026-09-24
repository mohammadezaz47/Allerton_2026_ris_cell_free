# ITNG YMYFA 80asra YA
# IMZZ

# src/channels.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

from .config import Config
from .correlation_setups import CorrelationSetups
from .large_scale import LargeScaleSetups
from .ris_phase_shifts import PhaseShiftSetups


@dataclass(frozen=True)
class ChannelBlock:
    """
    One block of channel realizations for one setup.

    Shapes
    - h_ap_user: (L, M, K, Rb)
    - H_ap_ris:  (L, T, N, M, Rb)
    - g_ris_user:(T, N, K, Rb)
    - u_ap_user: (L, M, K, Rb)  effective channel u = h + sum_t H^H Theta g
    """
    setup_index: int
    r0: int
    r1: int

    h_ap_user: np.ndarray
    H_ap_ris: np.ndarray
    g_ris_user: np.ndarray
    u_ap_user: np.ndarray


@dataclass(frozen=True)
class ChannelSetups:
    """
    Full stored channel realizations for all setups.

    Shapes
    - h_ap_user: (S, L, M, K, R)
    - H_ap_ris:  (S, L, T, N, M, R)
    - g_ris_user:(S, T, N, K, R)
    - u_ap_user: (S, L, M, K, R)
    """
    h_ap_user: np.ndarray
    H_ap_ris: np.ndarray
    g_ris_user: np.ndarray
    u_ap_user: np.ndarray

    is_blocked_ap_user: np.ndarray  # (S, L, K)


def _complex_gaussian(shape: tuple[int, ...], rng: np.random.Generator, dtype: np.dtype) -> np.ndarray:
    """
    i.i.d. CN(0,1) samples.

    Each entry is (X + jY)/sqrt(2), X,Y ~ N(0,1).
    """
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    out = (real + 1j * imag) / np.sqrt(2.0)
    return out.astype(dtype, copy=False)


def sqrtm_hermitian_psd(R: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Matrix square-root for a Hermitian PSD matrix.

    Returns L such that L @ L^H ≈ R.
    """
    R = np.asarray(R, dtype=np.complex128)
    R = 0.5 * (R + R.conj().T)

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_eigvals = np.sqrt(eigvals)

    L = (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.conj().T
    return L.astype(dtype, copy=False)


def sqrtm_hermitian_psd_stack(R: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Batched matrix square-root for Hermitian PSD matrices.

    Input shape
      (..., n, n)

    Output shape
      (..., n, n) such that L @ L^H ≈ R
    """
    R = np.asarray(R, dtype=np.complex128)
    R = 0.5 * (R + np.swapaxes(R.conj(), -1, -2))

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 0.0, None)
    sqrt_eigvals = np.sqrt(eigvals)

    V_scaled = eigvecs * sqrt_eigvals[..., None, :]
    L = V_scaled @ np.swapaxes(eigvecs.conj(), -1, -2)
    return L.astype(dtype, copy=False)


def iter_channel_blocks(
    cfg: Config,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    phases: Optional[PhaseShiftSetups],
    seed: Optional[int] = None,
    dtype: np.dtype = np.complex64,
    block_len: int = 128,
) -> Generator[ChannelBlock, None, None]:
    """
    Yield channel realizations in blocks.

    Blockage consistency
    - No extra blockage mask is applied here.
    - Blockage already affected ls.ap_user_beta_over_noise, so channels are weakened or zero.

    No-RIS consistency
    - If RIS is disabled, u_ap_user equals h_ap_user and RIS arrays are empty.
    """
    cfg.validate()

    if block_len <= 0:
        raise ValueError("block_len must be positive")

    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    S = int(cfg.sim.num_setups)
    R = int(cfg.sim.num_realizations)

    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)

    if corr.delta_ap_user.shape != (S, L, K, M, M):
        raise ValueError("corr.delta_ap_user must have shape (S, L, K, M, M)")

    T = int(corr.delta_ap_ris.shape[2])
    N = int(corr.R_ris.shape[0]) if T > 0 else 0

    ris_active = (cfg.ris.enable_ris and T > 0 and N > 0)

    if ris_active:
        if phases is None:
            raise ValueError("phases must be provided when RIS is enabled")

        if phases.vartheta.shape[0] != S:
            raise ValueError("phases.vartheta must have shape (S, N)")
        if phases.vartheta.shape[1] != N:
            raise ValueError("phases.vartheta second dimension must match N from corr.R_ris")

        if corr.delta_ap_ris.shape != (S, L, T, M, M):
            raise ValueError("corr.delta_ap_ris must have shape (S, L, T, M, M)")
        if corr.R_ris.shape != (N, N):
            raise ValueError("corr.R_ris must have shape (N, N)")
        if ls.ap_ris_beta_over_noise.shape != (S, L, T):
            raise ValueError("ls.ap_ris_beta_over_noise must have shape (S, L, T)")
        if ls.ris_user_beta_over_noise.shape != (S, T, K):
            raise ValueError("ls.ris_user_beta_over_noise must have shape (S, T, K)")

    R_ris_sqrt = sqrtm_hermitian_psd(corr.R_ris, dtype=dtype) if ris_active else None

    for s in range(S):
        Delta_user_sqrt = sqrtm_hermitian_psd_stack(corr.delta_ap_user[s], dtype=dtype)  # (L, K, M, M)
        beta_user_sqrt = np.sqrt(np.asarray(ls.ap_user_beta_over_noise[s], dtype=float))  # (L, K)

        if ris_active:
            Delta_ap_ris_sqrt = sqrtm_hermitian_psd_stack(corr.delta_ap_ris[s], dtype=dtype)  # (L, T, M, M)
            beta_ap_ris_sqrt = np.sqrt(np.asarray(ls.ap_ris_beta_over_noise[s], dtype=float))  # (L, T)
            beta_ris_user_sqrt = np.sqrt(np.asarray(ls.ris_user_beta_over_noise[s], dtype=float))  # (T, K)

            d = np.exp(1j * np.asarray(phases.vartheta[s], dtype=float)).astype(dtype, copy=False)  # (N,)

        for r0 in range(0, R, block_len):
            r1 = min(R, r0 + block_len)
            rb = r1 - r0

            # AP -> User
            W_user = _complex_gaussian((L, K, M, rb), rng=rng, dtype=dtype)
            h_lkmr = np.einsum("lkmn,lknr->lkmr", Delta_user_sqrt, W_user, optimize=True)  # (L, K, M, rb)
            h_lkmr *= beta_user_sqrt[..., None, None]
            h_block = np.transpose(h_lkmr, (0, 2, 1, 3))  # (L, M, K, rb)

            if ris_active:
                # RIS -> User
                W_g = _complex_gaussian((T, K, N, rb), rng=rng, dtype=dtype)
                g_tknr = np.einsum("nm,tknr->tkmr", R_ris_sqrt, W_g, optimize=True)  # (T, K, N, rb)
                g_tknr *= beta_ris_user_sqrt[..., None, None]
                g_block = np.transpose(g_tknr, (0, 2, 1, 3))  # (T, N, K, rb)

                # AP -> RIS
                W_H = _complex_gaussian((L, T, N, M, rb), rng=rng, dtype=dtype)

                tmp = np.einsum("ij,ltjmr->ltimr", R_ris_sqrt, W_H, optimize=True)  # (L, T, N, M, rb)
                tmp *= beta_ap_ris_sqrt[..., None, None, None]
                H_block = np.einsum("ltimr,ltmn->ltinr", tmp, Delta_ap_ris_sqrt, optimize=True)  # (L, T, N, M, rb)

                # Cascaded effective channel
                # Theta is diagonal, so Theta*g is elementwise multiply by d
                g_phase = g_block * d[None, :, None, None]  # (T, N, K, rb)

                # H^H Theta g
                # H_block is (L, T, N, M, rb) so conj(H) times g over N gives (L, T, M, K, rb)
                casc = np.einsum("ltnmr,tnkr->ltmkr", np.conj(H_block), g_phase, optimize=True)
                casc_sum = np.sum(casc, axis=1)  # sum over T, gives (L, M, K, rb)

                u_block = h_block + casc_sum
            else:
                H_block = np.empty((L, 0, 0, M, rb), dtype=dtype)
                g_block = np.empty((0, 0, K, rb), dtype=dtype)
                u_block = h_block

            yield ChannelBlock(
                setup_index=s,
                r0=r0,
                r1=r1,
                h_ap_user=h_block,
                H_ap_ris=H_block,
                g_ris_user=g_block,
                u_ap_user=u_block,
            )


def generate_channels_setups(
    cfg: Config,
    corr: CorrelationSetups,
    ls: LargeScaleSetups,
    phases: Optional[PhaseShiftSetups],
    seed: Optional[int] = None,
    dtype: np.dtype = np.complex64,
    block_len: int = 128,
    store_ris_links: bool = False,
) -> ChannelSetups:
    """
    Materialize channels for all setups into full arrays.

    u_ap_user is always stored.
    H_ap_ris and g_ris_user are stored only if store_ris_links is True.
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)
    R = int(cfg.sim.num_realizations)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)
    M = int(cfg.dims.num_ap_antennas)

    T = int(corr.delta_ap_ris.shape[2])
    N = int(corr.R_ris.shape[0]) if T > 0 else 0
    ris_active = (cfg.ris.enable_ris and T > 0 and N > 0)

    h_all = np.zeros((S, L, M, K, R), dtype=dtype)
    u_all = np.zeros((S, L, M, K, R), dtype=dtype)

    if store_ris_links and ris_active:
        H_all = np.zeros((S, L, T, N, M, R), dtype=dtype)
        g_all = np.zeros((S, T, N, K, R), dtype=dtype)
    else:
        H_all = np.empty((S, L, 0, 0, M, R), dtype=dtype)
        g_all = np.empty((S, 0, 0, K, R), dtype=dtype)

    for blk in iter_channel_blocks(
        cfg=cfg,
        corr=corr,
        ls=ls,
        phases=phases,
        seed=seed,
        dtype=dtype,
        block_len=block_len,
    ):
        s = blk.setup_index
        r0, r1 = blk.r0, blk.r1

        h_all[s, :, :, :, r0:r1] = blk.h_ap_user
        u_all[s, :, :, :, r0:r1] = blk.u_ap_user

        if store_ris_links and ris_active:
            H_all[s, :, :, :, :, r0:r1] = blk.H_ap_ris
            g_all[s, :, :, :, r0:r1] = blk.g_ris_user

    return ChannelSetups(
        h_ap_user=h_all,
        H_ap_ris=H_all,
        g_ris_user=g_all,
        u_ap_user=u_all,
        is_blocked_ap_user=ls.is_blocked_ap_user,
    )