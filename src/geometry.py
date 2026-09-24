# ITNG YMYFA 80asra YA
# IMZZ

# src/geometry.py

"""
geometry.py

Drop node locations for a single setup or many setups.

Layout
- One square area
- One hotspot disk inside the square
- Users are dropped inside the hotspot disk
- APs are dropped inside the square but outside (hotspot radius + guard width)
- RIS is either at hotspot center or on hotspot edge

Wrap-around
- When enabled, distance computations use the minimum-image convention
- Dropping constraints that depend on distance (guard zone, RIS exclusion) also use wrap-around
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import Config



# ------- Data containers -------

@dataclass(frozen=True)
class DistanceCache:
    """Cached pairwise distances for one setup."""
    ap_user_2d: np.ndarray   # (L, K)
    ap_user_3d: np.ndarray   # (L, K)

    ap_ris_2d: np.ndarray    # (L, T)
    ap_ris_3d: np.ndarray    # (L, T)

    ris_user_2d: np.ndarray  # (T, K)
    ris_user_3d: np.ndarray  # (T, K)



@dataclass(frozen=True)
class SetupGeometry:
    """Geometry for one setup."""
    ap_xy: np.ndarray               # (L, 2)
    user_xy: np.ndarray             # (K, 2)
    ris_xy: np.ndarray              # (T, 2) where T is 0 or num_ris
    area_bounds: Tuple[float, float, float, float]  #(x_min, x_max, y_min, y_max)
    dist: DistanceCache



@dataclass(frozen=True)
class DistanceCacheSetups:
    """Cached pairwise distances for all setups."""
    ap_user_2d: np.ndarray   # (S, L, K)
    ap_user_3d: np.ndarray   # (S, L, K)

    ap_ris_2d: np.ndarray    # (S, L, T)
    ap_ris_3d: np.ndarray    # (S, L, T)

    ris_user_2d: np.ndarray  # (S, T, K)
    ris_user_3d: np.ndarray  # (S, T, K)



@dataclass(frozen=True)
class GeometrySetups:
    """Geometry for all setups."""
    ap_xy: np.ndarray     # (S, L, 2)
    user_xy: np.ndarray   # (S, K, 2)
    ris_xy: np.ndarray    # (S, T, 2)
    area_bounds: Tuple[float, float, float, float]
    dist: DistanceCacheSetups



# ------- Core helpers -------

def _square_bounds(center_xy: Tuple[float, float], side_m: float) -> Tuple[float, float, float, float]:
    cx, cy = center_xy
    half = 0.5 * side_m
    x_min = cx - half
    x_max = cx + half
    y_min = cy - half
    y_max = cy + half
    return x_min, x_max, y_min, y_max



def _min_image_delta(delta: np.ndarray, side_m: float) -> np.ndarray:
    """
    Minimum-image convention for a periodic square of side side_m.
    Works for any origin since it uses only deltas.
    """
    return delta - side_m * np.round(delta / side_m)



def deltas_xy(A: np.ndarray, B: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Pairwise 2D deltas A - B with optional wrap-around.

    A shape can be (..., 2)
    B shape can be (..., 2)
    Numpy broadcasting applies.
    """
    delta = A - B
    if cfg.geom.wraparound_enabled:
        delta = _min_image_delta(delta, cfg.geom.area_side_m)
    return delta



def distances_2d(A: np.ndarray, B: np.ndarray, cfg: Config) -> np.ndarray:
    """Pairwise 2D distances with optional wrap-around."""
    d = deltas_xy(A, B, cfg)
    return np.linalg.norm(d, axis=-1)



def _build_distance_cache(ap_xy: np.ndarray, user_xy: np.ndarray, ris_xy: np.ndarray, cfg: Config) -> DistanceCache:
    """
    Build and return all pairwise distances for one setup.
    All 2D distances use wrap-around if enabled.
    3D distances are computed by adding height differences.
    """
    # 2D distances
    ap_user_2d = distances_2d(ap_xy[:, None, :], user_xy[None, :, :], cfg)   # (L, K)
    ap_ris_2d = distances_2d(ap_xy[:, None, :], ris_xy[None, :, :], cfg)     # (L, T)
    ris_user_2d = distances_2d(ris_xy[:, None, :], user_xy[None, :, :], cfg) # (T, K)

    # Height differences
    dh_ap_user = cfg.heights.ap_height_m - cfg.heights.user_height_m
    dh_ap_ris = cfg.heights.ap_height_m - cfg.heights.ris_height_m
    dh_ris_user = cfg.heights.ris_height_m - cfg.heights.user_height_m

    # 3D distances
    ap_user_3d = np.sqrt(ap_user_2d * ap_user_2d + dh_ap_user * dh_ap_user)
    ap_ris_3d = np.sqrt(ap_ris_2d * ap_ris_2d + dh_ap_ris * dh_ap_ris)
    ris_user_3d = np.sqrt(ris_user_2d * ris_user_2d + dh_ris_user * dh_ris_user)

    return DistanceCache(
        ap_user_2d=ap_user_2d,
        ap_user_3d=ap_user_3d,
        ap_ris_2d=ap_ris_2d,
        ap_ris_3d=ap_ris_3d,
        ris_user_2d=ris_user_2d,
        ris_user_3d=ris_user_3d,
    )



def distances_3d_ap_user(ap_xy: np.ndarray, user_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """
    3D distances between APs and users.
    Returns a matrix with shape (L, K)
    """
    d2 = distances_2d(ap_xy[:, None, :], user_xy[None, :, :], cfg)
    dh = cfg.heights.ap_height_m - cfg.heights.user_height_m
    return np.sqrt(d2 * d2 + dh * dh)



def distances_3d_ap_ris(ap_xy: np.ndarray, ris_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """
    3D distances between APs and RIS panels.
    Returns a matrix with shape (L, T).
    """
    if ris_xy.shape[0] == 0:
        return np.empty((ap_xy.shape[0], 0), dtype=float)
    d2 = distances_2d(ap_xy[:, None, :], ris_xy[None, :, :], cfg)
    dh = cfg.heights.ap_height_m - cfg.heights.ris_height_m
    return np.sqrt(d2 * d2 + dh * dh)



def distances_3d_ris_user(ris_xy: np.ndarray, user_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """
    3D distances between RIS panels and users.
    Returns a matrix with shape (T, K).
    """
    if ris_xy.shape[0] == 0:
        return np.empty((0, user_xy.shape[0]), dtype=float)
    d2 = distances_2d(ris_xy[:, None, :], user_xy[None, :, :], cfg)
    dh = cfg.heights.ris_height_m - cfg.heights.user_height_m
    return np.sqrt(d2 * d2 + dh * dh)



# ------- Dropping logic -------

def _drop_ris_xy(cfg: Config) -> np.ndarray:
    """
    Returns ris_xy with shape (T, 2).
    T is 0 if RIS is disabled or num_ris is 0.
    """
    if (not cfg.ris.enable_ris) or (cfg.dims.num_ris == 0):
        return np.empty((0, 2), dtype=float)
    
    center = np.array(cfg.geom.hotspot_center_xy, dtype=float)
    if cfg.geom.ris_placement == "center":
        base = center
    elif cfg.geom.ris_placement == "edge":
        r = cfg.geom.hotspot_radius_m - cfg.geom.ris_edge_inset_m
        if r < 0:
            raise ValueError("ris_edge_inset_m is larger than hotspot_radius_m")
        theta = np.deg2rad(cfg.geom.ris_edge_angle_deg)
        base = center + r * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    else:
        raise ValueError(f"Unknown ris_placement {cfg.geom.ris_placement}")

    # If you later set num_ris > 1, this repeats the same placement.
    return np.repeat(base[None, :], repeats=cfg.dims.num_ris, axis=0)



def _sample_uniform_in_disk(
    rng: np.random.Generator,
    center_xy: np.ndarray,
    radius_m: float,
    n: int,
) -> np.ndarray:
    """
    Uniform samples in a disk.
    r uses sqrt(U) so points are uniform by area, not by radius.
    """
    u = rng.random(n)
    v = rng.random(n)
    r = radius_m * np.sqrt(u)
    theta = 2.0 * np.pi * v
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return center_xy[None, :] + xy



def _drop_users_xy(cfg: Config, rng: np.random.Generator, ris_xy: np.ndarray) -> np.ndarray:
    """
    Drop K users inside the hotspot disk.
    Reject samples that fall inside the RIS exclusion radius.
    """
    K = cfg.dims.num_users
    center = np.array(cfg.geom.hotspot_center_xy, dtype=float)

    max_tries = max(10_000, 50 * K)
    accepted: list[np.ndarray] = []

    tries = 0
    batch_size = max(256, 20 * K)

    while len(accepted) < K and tries < max_tries:
        tries += 1

        batch = _sample_uniform_in_disk(rng, center, cfg.geom.hotspot_radius_m, batch_size)

        if ris_xy.shape[0] > 0 and cfg.geom.ris_user_exclusion_radius_m > 0:
            d_ris = distances_2d(batch[:, None, :], ris_xy[None, :, :], cfg)  # (B, T)
            ok = np.all(d_ris >= cfg.geom.ris_user_exclusion_radius_m, axis=1)
            batch = batch[ok]

        for p in batch:
            accepted.append(p)
            if len(accepted) == K:
                break

    if len(accepted) < K:
        raise RuntimeError("Failed to drop enough users. Relax ris_user_exclusion_radius_m or increase hotspot_radius_m")

    return np.stack(accepted, axis=0)



def _ap_region_mask(points_xy: np.ndarray, cfg: Config) -> np.ndarray:
    mode = getattr(cfg.geom, "ap_region_mode", "full_area")

    if mode == "full_area":
        return np.ones(points_xy.shape[0], dtype=bool)

    if mode == "rectangle":
        x_min, x_max, y_min, y_max = cfg.geom.ap_region_bounds_xyxy
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        return (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    raise ValueError(f"Unknown ap_region_mode {mode}")



def _outside_guard_zone_mask(points_xy: np.ndarray, cfg: Config) -> np.ndarray:
    """
    True for points that are outside the forbidden disk around the hotspot center.
    Forbidden radius is hotspot_radius + guard_width.
    """
    center = np.array(cfg.geom.hotspot_center_xy, dtype=float)
    forbidden_r = cfg.geom.hotspot_radius_m + cfg.geom.guard_width_m
    d = distances_2d(points_xy, center[None, :], cfg)
    return d > forbidden_r



def _drop_aps_random(cfg: Config, rng: np.random.Generator, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """Random AP drop in the square, outside hotspot plus guard zone."""
    L = cfg.dims.num_aps
    x_min, x_max, y_min, y_max = bounds

    min_sep = float(cfg.geom.ap_min_seperation_m)
    max_tries = int(cfg.geom.ap_drop_max_tries)

    accepted: list[np.ndarray] = []
    tries = 0

    batch_size = max(512, 40 * L)

    while len(accepted) < L and tries < max_tries:
        tries += 1

        batch = rng.uniform(low=(x_min, y_min), high=(x_max, y_max), size=(batch_size, 2))

        mask_guard = _outside_guard_zone_mask(batch, cfg)
        mask_region = _ap_region_mask(batch, cfg)
        batch = batch[mask_guard & mask_region]

        for cand in batch:
            if min_sep > 0 and len(accepted) > 0:
                aps = np.stack(accepted, axis=0)
                d = distances_2d(aps, cand[None, :], cfg)
                if np.min(d) < min_sep:
                    continue

            accepted.append(cand)
            if len(accepted) == L:
                break

    if len(accepted) < L:
        raise RuntimeError("Failed to drop enough APs. Relax guard_width_m or ap_min_seperation_m")

    return np.stack(accepted, axis=0)



def _drop_aps_grid(cfg: Config, rng: np.random.Generator, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    """Grid AP drop in the square, then select L valid points."""
    L = cfg.dims.num_aps
    x_min, x_max, y_min, y_max = bounds

    spacing = float(cfg.geom.grid_spacing_m)
    off = np.array(cfg.geom.grid_offset_xy, dtype=float)

    x0 = x_min + off[0]
    y0 = y_min + off[1]

    xs = np.arange(x0, x_max + 1e-12, spacing)
    ys = np.arange(y0, y_max + 1e-12, spacing)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    pts = pts[_outside_guard_zone_mask(pts, cfg)]
    pts = pts[_ap_region_mask(pts, cfg)]

    if pts.shape[0] < L:
        raise RuntimeError("Grid produced fewer valid AP points than num_aps. Decrease grid_spacing_m or guard_width_m")

    idx = rng.choice(pts.shape[0], size=L, replace=False)
    return pts[idx]



def drop_setup_geometry(cfg: Config, rng: np.random.Generator) -> SetupGeometry:
    """
    Drop AP, user, and RIS locations for one setup.
    """
    bounds = _square_bounds(cfg.geom.area_center_xy, cfg.geom.area_side_m)

    ris_xy = _drop_ris_xy(cfg)

    user_xy = _drop_users_xy(cfg, rng, ris_xy)

    if cfg.geom.ap_placement == "random":
        ap_xy = _drop_aps_random(cfg, rng, bounds)
    elif cfg.geom.ap_placement == "grid":
        ap_xy = _drop_aps_grid(cfg, rng, bounds)
    else:
        raise ValueError(f"Unknown ap_placement {cfg.geom.ap_placement}")

    dist = _build_distance_cache(ap_xy=ap_xy, user_xy=user_xy, ris_xy=ris_xy, cfg=cfg)
    return SetupGeometry(ap_xy=ap_xy, user_xy=user_xy, ris_xy=ris_xy, area_bounds=bounds, dist=dist)



def generate_geometry_setups(cfg: Config) -> GeometrySetups:
    """
    Drop geometry for all setups using cfg.sim.seed.
    Returns arrays stacked over setups.
    """
    rng = np.random.default_rng(cfg.sim.seed)

    S = cfg.sim.num_setups
    L = cfg.dims.num_aps
    K = cfg.dims.num_users
    T = cfg.dims.num_ris if cfg.ris.enable_ris else 0

    bounds = _square_bounds(cfg.geom.area_center_xy, cfg.geom.area_side_m)

    ap_all = np.empty((S, L, 2), dtype=float)
    user_all = np.empty((S, K, 2), dtype=float)
    ris_all = np.empty((S, T, 2), dtype=float)

    # Distance caches
    ap_user_2d_all = np.empty((S, L, K), dtype=float)
    ap_user_3d_all = np.empty((S, L, K), dtype=float)

    ap_ris_2d_all = np.empty((S, L, T), dtype=float)
    ap_ris_3d_all = np.empty((S, L, T), dtype=float)

    ris_user_2d_all = np.empty((S, T, K), dtype=float)
    ris_user_3d_all = np.empty((S, T, K), dtype=float)

    for s in range(S):
        g = drop_setup_geometry(cfg, rng)

        ap_all[s] = g.ap_xy
        user_all[s] = g.user_xy
        ris_all[s] = g.ris_xy

        ap_user_2d_all[s] = g.dist.ap_user_2d
        ap_user_3d_all[s] = g.dist.ap_user_3d

        ap_ris_2d_all[s] = g.dist.ap_ris_2d
        ap_ris_3d_all[s] = g.dist.ap_ris_3d

        ris_user_2d_all[s] = g.dist.ris_user_2d
        ris_user_3d_all[s] = g.dist.ris_user_3d

    dist_all = DistanceCacheSetups(
        ap_user_2d=ap_user_2d_all,
        ap_user_3d=ap_user_3d_all,
        ap_ris_2d=ap_ris_2d_all,
        ap_ris_3d=ap_ris_3d_all,
        ris_user_2d=ris_user_2d_all,
        ris_user_3d=ris_user_3d_all,
    )

    return GeometrySetups(
        ap_xy=ap_all,
        user_xy=user_all,
        ris_xy=ris_all,
        area_bounds=bounds,
        dist=dist_all,
    )