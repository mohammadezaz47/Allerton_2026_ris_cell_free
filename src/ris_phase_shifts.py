# ITNG YMYFA 80asra YA
# IMZZ

# src/ris_phase_shifts.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .config import Config


PhaseShiftMode = Literal["random", "equal"]



@dataclass(frozen=True)
class PhaseShiftSetups:
    """
    RIS phase shifts for all setups.

    Theta[s] is the RIS phase shift matrix for setup s:
      Theta[s] = diag(exp(1j * vartheta))

    Shapes
    - Theta: (S, N, N)
    - vartheta: (S, N)
    """
    Theta: np.ndarray
    vartheta: np.ndarray
    mode: PhaseShiftMode


def _diag_from_angles(vartheta: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Build Theta = diag(exp(j*vartheta)) for a batch of setups.

    vartheta shape: (S, N)
    output Theta shape: (S, N, N)
    """
    phases = np.exp(1j * vartheta).astype(dtype, copy=False)  # (S, N)
    S, N = phases.shape
    Theta = np.zeros((S, N, N), dtype=dtype)
    idx = np.arange(N)
    Theta[:, idx, idx] = phases
    return Theta



def build_ris_phase_shifts_setups(
    cfg: Config,
    mode: PhaseShiftMode = "random",
    seed: Optional[int] = None,
    equal_phase: float = 0.0,
    dtype: np.dtype = np.complex64,
) -> PhaseShiftSetups:
    """
    Build RIS phase shift matrices for all setups.

    Mathematical definition
      Theta = diag([e^{j*vartheta_1}, ..., e^{j*vartheta_N}]^T)
      vartheta_n in [-pi, pi]

    Modes
    - random: vartheta_n ~ Uniform[-pi, pi]
    - equal:  vartheta_n = equal_phase for all n
    """
    cfg.validate()

    S = int(cfg.sim.num_setups)

    # No-RIS consistent behavior
    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        Theta = np.empty((S, 0, 0), dtype=dtype)
        vartheta = np.empty((S, 0), dtype=float)
        return PhaseShiftSetups(Theta=Theta, vartheta=vartheta, mode=mode)

    N = int(cfg.dims.num_ris_elements)

    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)

    if mode == "random":
        vartheta = rng.uniform(low=-np.pi, high=np.pi, size=(S, N)).astype(float, copy=False)
        Theta = _diag_from_angles(vartheta, dtype=dtype)

    elif mode == "equal":
        vartheta = np.full((S, N), float(equal_phase), dtype=float)
        Theta = _diag_from_angles(vartheta, dtype=dtype)

    else:
        raise ValueError(f"Unknown mode {mode}")

    return PhaseShiftSetups(Theta=Theta, vartheta=vartheta, mode=mode)
