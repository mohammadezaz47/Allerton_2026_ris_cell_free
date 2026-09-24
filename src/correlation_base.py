# ITNG YMYFA 80asra YA
# IMZZ

# src/correlation_base.py

from __future__ import annotations

import numpy as np



def trace_normalize(R: np.ndarray, target_trace: float) -> np.ndarray:
    """
    Scale a square matrix so its trace equals target_trace.
    """
    R = np.asarray(R)
    tr = np.trace(R)

    if np.isclose(tr, 0.0):
        raise ValueError("Cannot trace-normalize a matrix with near-zero trace.")

    return R * (float(target_trace) / tr)



def ris_correlation_matrix(
    num_h: int,
    num_v: int,
    d_h_wavelengths: float = 0.5,
    d_v_wavelengths: float = 0.5,
    wavelength_m: float = 1.0,
) -> np.ndarray:
    """
    RIS-side spatial correlation matrix R (N×N) for a rectangular RIS grid.

    Model
      [R]_{mn} = sinc( 2 * ||q_m - q_n|| / lambda )
    where np.sinc(x) = sin(pi x)/(pi x).
    """
    num_h = int(num_h)
    num_v = int(num_v)
    if num_h <= 0 or num_v <= 0:
        raise ValueError("num_h and num_v must be positive integers.")

    d_h_wavelengths = float(d_h_wavelengths)
    d_v_wavelengths = float(d_v_wavelengths)
    if d_h_wavelengths <= 0 or d_v_wavelengths <= 0:
        raise ValueError("Element spacings must be positive.")

    wavelength_m = float(wavelength_m)
    if wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive.")

    d_h_m = d_h_wavelengths * wavelength_m
    d_v_m = d_v_wavelengths * wavelength_m

    i = np.arange(num_h, dtype=float)
    j = np.arange(num_v, dtype=float)

    X, Y = np.meshgrid(i * d_h_m, j * d_v_m, indexing="xy")

    xs = X.ravel()
    ys = Y.ravel()

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_m = np.sqrt(dx * dx + dy * dy)

    x = 2.0 * dist_m / wavelength_m
    R = np.sinc(x)

    R = 0.5 * (R + R.T)
    return R.real



def _hermitian_toeplitz_from_first_col(first_col: np.ndarray) -> np.ndarray:
    """
    Build an M×M Hermitian Toeplitz matrix from its first column.

    first_col[k] corresponds to correlation at lag k >= 0.
    """
    first_col = np.asarray(first_col, dtype=np.complex128)
    M = first_col.shape[0]

    idx = np.abs(np.arange(M)[:, None] - np.arange(M)[None, :])
    R = first_col[idx].copy()

    upper = np.arange(M)[:, None] < np.arange(M)[None, :]
    R[upper] = np.conj(R[upper])
    return R



def local_scattering_correlation(
    M: int,
    theta: float,
    asd_deg: float,
    antenna_spacing: float = 0.5,
    distribution: str = "gaussian",
    num_points: int = 4001,
) -> np.ndarray:
    """
    AP-side spatial correlation matrix (M×M) under a local scattering model.

    This is a vectorized version that computes the first column for all antenna
    separations at once, then builds a Hermitian Toeplitz matrix.

    Parameters
    ----------
    M
        Number of antennas at the AP (ULA).
    theta
        Nominal angle in radians.
    asd_deg
        Angular standard deviation in degrees.
    antenna_spacing
        Antenna spacing in wavelengths.
    distribution
        'gaussian', 'uniform', or 'laplace'.
    num_points
        Integration resolution.
    """
    M = int(M)
    if M <= 0:
        raise ValueError("M must be positive.")

    asd_rad = float(asd_deg) * np.pi / 180.0
    if asd_rad <= 0:
        raise ValueError("asd_deg must be positive.")

    antenna_spacing = float(antenna_spacing)
    if antenna_spacing <= 0:
        raise ValueError("antenna_spacing must be positive.")

    dist_type = distribution.lower()
    if dist_type not in ("gaussian", "uniform", "laplace"):
        raise ValueError('distribution must be "gaussian", "uniform", or "laplace".')

    num_points = int(num_points)
    if num_points < 101:
        raise ValueError("num_points should be at least 101 for reasonable accuracy.")

    # Integration grid over delta
    if dist_type == "gaussian":
        limit = 20.0 * asd_rad
        delta = np.linspace(-limit, limit, num_points)
    elif dist_type == "uniform":
        U = np.sqrt(3.0) * asd_rad
        delta = np.linspace(-U, U, num_points)
    else:
        limit = 20.0 * asd_rad
        delta = np.linspace(-limit, limit, num_points)

    # PDF on the grid
    if dist_type == "gaussian":
        pdf = np.exp(-delta * delta / (2.0 * asd_rad * asd_rad)) / (np.sqrt(2.0 * np.pi) * asd_rad)
    elif dist_type == "uniform":
        U = np.sqrt(3.0) * asd_rad
        pdf = np.ones_like(delta) / (2.0 * U)
    else:
        b = asd_rad / np.sqrt(2.0)
        pdf = np.exp(-np.abs(delta) / b) / (2.0 * b)

    two_pi = 2.0 * np.pi
    theta = float(theta)

    # Vector of antenna separations in wavelengths
    m = np.arange(M, dtype=float)                 # (M,)
    d = antenna_spacing * m                       # (M,)

    # phase has shape (M, P) where P=num_points
    phase = two_pi * d[:, None] * np.sin(theta + delta[None, :])
    exp_term = np.exp(1j * phase)

    integrand = exp_term * pdf[None, :]
    first_col = np.trapezoid(integrand, delta, axis=1)  # (M,)

    return _hermitian_toeplitz_from_first_col(first_col)