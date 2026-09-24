# ITNG YMYFA 80asra YA
# IMZZ

# src/simulation_runner.py

from __future__ import annotations

from dataclasses import dataclass, replace, is_dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np

from .config import Config, make_default_config
from .geometry import GeometrySetups, generate_geometry_setups
from .large_scale import LargeScaleSetups, compute_large_scale_setups
from .correlation_setups import CorrelationSetups, build_correlation_setups
from .ris_phase_shifts import PhaseShiftSetups, build_ris_phase_shifts_setups
from .pilot_assignment import PilotAssignmentSetups, assign_pilots_setups
from .power_allocation import (
      PowerAllocationSetups,
      OptimizedPowerOptions,
      allocate_downlink_power,
)
from .channels import ChannelSetups, generate_channels_setups
from .channel_estimation import (
    ChannelEstimationSetups,
    build_channel_estimation_statistics,
    estimate_u_mmse_realizations,
)
from .dl_sinr_monte_carlo import DownlinkSinrMonteCarloSetups, compute_dl_sinr_monte_carlo
from .dl_sinr_closed_form import DownlinkSinrClosedFormSetups, compute_dl_sinr_closed_form
from .spectral_efficiency import SpectralEfficiencySetups, compute_spectral_efficiency_setups
from .energy_efficiency import EnergyEfficiencySetups, compute_energy_efficiency_setups


SinrMode = Literal["closed_form", "monte_carlo", "both"]
ServingMode = Literal["candidate", "conventional", "custom"]
PilotScheme = Literal["heuristic", "random"]
PhaseMode = Literal["random", "equal", "optimized"]
PowerScheme = Literal["equal", "optimized"]



@dataclass(frozen=True)
class PreparedUpstream:
    """
    Upstream pipeline objects that can be reused across runs.

    This is useful for apples-to-apples comparisons and downstream sweeps.
    """
    cfg: Config
    geom: GeometrySetups
    large_scale: LargeScaleSetups
    corr: CorrelationSetups
    phases: PhaseShiftSetups




@dataclass(frozen=True)
class SimulationResults:
    """
    Full results for one experiment run.
    """
    cfg: Config
    prepared_upstream: PreparedUpstream

    pilot: PilotAssignmentSetups
    serve_mask_used: np.ndarray
    power: PowerAllocationSetups

    channels: ChannelSetups | None
    est_stats: ChannelEstimationSetups
    est_full: ChannelEstimationSetups | None

    sinr_closed_form: DownlinkSinrClosedFormSetups | None
    sinr_monte_carlo: DownlinkSinrMonteCarloSetups | None

    se_closed_form: SpectralEfficiencySetups | None
    se_monte_carlo: SpectralEfficiencySetups | None

    ee_closed_form: EnergyEfficiencySetups | None
    ee_monte_carlo: EnergyEfficiencySetups | None

    summary: dict[str, float | int | None]



@dataclass(frozen=True)
class SweepResults:
    """
    Results from sweeping one parameter
    """
    sweep_name: str
    sweep_values: list[Any]
    results: list[SimulationResults]
    summary_rows: list[dict[str, float | int | None]]



#------- Small helpers -------

def _stage_seeds(base_seed: int) -> dict[str, int]:
    """
    Give each stage a deterministic but different seed.
    
    This avoids accidental coupling across stages while keeping everything reporducible
    """
    return {
        "geometry": base_seed + 101,
        "large_scale": base_seed + 202,
        "phase": base_seed + 303,
        "pilot": base_seed + 404,
        "channels": base_seed + 505,
        "estimation": base_seed + 606,
    }



def _set_nested_dataclass_value(obj: Any, parts: list[str], value: Any) -> Any:
    """
    Recursively update a frozen nested dataclass using dataclasses.replace.

    Example
    - parts = ["pilots", "pilot_len"]
    """
    if not is_dataclass(obj):
        raise TypeError("Expected a dataclass object while applying overrides")

    if len(parts) == 1:
        return replace(obj, **{parts[0]: value})

    head = parts[0]
    tail = parts[1:]

    child = getattr(obj, head)
    updated_child = _set_nested_dataclass_value(child, tail, value)
    return replace(obj, **{head: updated_child})



def apply_config_overrides(cfg: Config, overrides: Optional[dict[str, Any]]) -> Config:
    """
    Apply dot-path overrides to the frozen config dataclass.
    
    Example
    - {"pilots.pilot_len": 6, "sim.num_setups": 20}
    """
    if overrides is None or len(overrides) == 0:
        cfg.validate()
        return cfg

    out = cfg
    for key, value in overrides.items():
        parts = key.split(".")
        out = _set_nested_dataclass_value(out, parts, value)

    out.validate()
    return out



def _overrides_touch_upstream(overrides: Optional[dict[str, Any]]) -> bool:
    """
    Decide whether a set of overrides changes upstream objects.

    Upstream means anything that affects
    - geometry
    - large-scale
    - correlations
    - RIS phases / dimensions
    """
    if not overrides:
        return False

    upstream_prefixes = (
        "dims.",
        "geom.",
        "heights.",
        "prop.",
        "arrays.",
        "fading.",
        "noise.",
        "ris.",
        "sim.num_setups",
        "sim.seed",
    )

    for key in overrides.keys():
        if key == "sim.num_setups" or key == "sim.seed":
            return True
        if key.startswith(upstream_prefixes):
            return True

    return False



def _build_custom_mask_from_users(cfg: Config, serve_users: Sequence[Sequence[Sequence[int]]]) -> np.ndarray:
    """
    Convert serve_users[s][l] into a boolean serving mask with shape (S, L, K).
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    if len(serve_users) != S:
        raise ValueError("custom_serve_users must have length S")

    mask = np.zeros((S, L, K), dtype=bool)

    for s in range(S):
        if len(serve_users[s]) != L:
            raise ValueError("custom_serve_users[s] must have length L")
        for l in range(L):
            users_l = np.asarray(serve_users[s][l], dtype=int).ravel()
            if users_l.size == 0:
                continue
            if np.any(users_l < 0) or np.any(users_l >= K):
                raise ValueError("custom_serve_users contains user indices out of range")
            mask[s, l, users_l] = True

    return mask



def _resolve_serving_mask(
    cfg: Config,
    serving_mode: ServingMode,
    pilot_out: PilotAssignmentSetups,
    custom_serve_mask: Optional[np.ndarray],
    custom_serve_users: Optional[Sequence[Sequence[Sequence[int]]]],
) -> np.ndarray:
    """
    Resolve the serving set used by downstream modules.
    """
    S = int(cfg.sim.num_setups)
    L = int(cfg.dims.num_aps)
    K = int(cfg.dims.num_users)

    if serving_mode == "candidate":
        return np.asarray(pilot_out.candidate_mask, dtype=bool)

    if serving_mode == "conventional":
        return np.ones((S, L, K), dtype=bool)

    if serving_mode == "custom":
        if custom_serve_mask is not None and custom_serve_users is not None:
            raise ValueError("Provide either custom_serve_mask or custom_serve_users, not both")

        if custom_serve_mask is not None:
            m = np.asarray(custom_serve_mask)
            if m.shape != (S, L, K):
                raise ValueError("custom_serve_mask must have shape (S, L, K)")
            return m.astype(bool, copy=False)

        if custom_serve_users is not None:
            return _build_custom_mask_from_users(cfg, custom_serve_users)

        raise ValueError("serving_mode='custom' requires custom_serve_mask or custom_serve_users")

    raise ValueError(f"Unknown serving_mode {serving_mode}")



def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(np.asarray(x, dtype=float)))



def _build_summary(
    serve_mask_used: np.ndarray,
    sinr_cf: DownlinkSinrClosedFormSetups | None,
    sinr_mc: DownlinkSinrMonteCarloSetups | None,
    se_cf: SpectralEfficiencySetups | None,
    se_mc: SpectralEfficiencySetups | None,
    ee_cf: EnergyEfficiencySetups | None,
    ee_mc: EnergyEfficiencySetups | None,
) -> dict[str, float | int | None]:
    """
    Build compact setup-averaged metrics for plotting and quick checks.
    """
    S, L, K = serve_mask_used.shape

    active_aps_per_setup = np.sum(np.any(serve_mask_used, axis=2), axis=1)
    serving_aps_per_user = np.sum(serve_mask_used, axis=1)

    summary: dict[str, float | int | None] = {
        "num_setups": S,
        "num_aps": L,
        "num_users": K,
        "avg_active_aps_per_setup": _safe_mean(active_aps_per_setup),
        "avg_serving_aps_per_user": _safe_mean(serving_aps_per_user),
    }

    if sinr_cf is not None:
        summary["mean_sinr_cf"] = _safe_mean(sinr_cf.sinr_dl)
        summary["mean_ds_cf"] = _safe_mean(sinr_cf.ds_power)
        summary["mean_uu_cf"] = _safe_mean(sinr_cf.uu_power)
        summary["mean_ci_cf"] = _safe_mean(sinr_cf.ci_power)
        summary["mean_ni_cf"] = _safe_mean(sinr_cf.ni_power)
    else:
        summary["mean_sinr_cf"] = None

    if sinr_mc is not None:
        summary["mean_sinr_mc"] = _safe_mean(sinr_mc.sinr_dl)
        summary["mean_ds_mc"] = _safe_mean(sinr_mc.ds_power)
        summary["mean_uu_mc"] = _safe_mean(sinr_mc.uu_power)
        summary["mean_ci_mc"] = _safe_mean(sinr_mc.ci_power)
        summary["mean_ni_mc"] = _safe_mean(sinr_mc.ni_power)
    else:
        summary["mean_sinr_mc"] = None

    if se_cf is not None:
        summary["mean_se_cf"] = _safe_mean(se_cf.se)
        summary["mean_sum_se_cf"] = _safe_mean(se_cf.sum_se)
    else:
        summary["mean_se_cf"] = None

    if se_mc is not None:
        summary["mean_se_mc"] = _safe_mean(se_mc.se)
        summary["mean_sum_se_mc"] = _safe_mean(se_mc.sum_se)
    else:
        summary["mean_se_mc"] = None

    if ee_cf is not None:
        summary["mean_ee_cf_bit_per_joule"] = _safe_mean(ee_cf.ee_bit_per_joule)
        summary["mean_ee_cf_mbit_per_joule"] = _safe_mean(ee_cf.ee_mbit_per_joule)
    else:
        summary["mean_ee_cf_bit_per_joule"] = None

    if ee_mc is not None:
        summary["mean_ee_mc_bit_per_joule"] = _safe_mean(ee_mc.ee_bit_per_joule)
        summary["mean_ee_mc_mbit_per_joule"] = _safe_mean(ee_mc.ee_mbit_per_joule)
    else:
        summary["mean_ee_mc_bit_per_joule"] = None

    if sinr_cf is not None and sinr_mc is not None:
        diff = np.asarray(sinr_cf.sinr_dl, dtype=float) - np.asarray(sinr_mc.sinr_dl, dtype=float)
        summary["mean_abs_sinr_gap"] = _safe_mean(np.abs(diff))
        summary["mean_rel_sinr_gap"] = _safe_mean(np.abs(diff) / np.maximum(np.abs(sinr_mc.sinr_dl), 1e-12))

    if se_cf is not None and se_mc is not None:
        diff = np.asarray(se_cf.se, dtype=float) - np.asarray(se_mc.se, dtype=float)
        summary["mean_abs_se_gap"] = _safe_mean(np.abs(diff))

    if ee_cf is not None and ee_mc is not None:
        diff = np.asarray(ee_cf.ee_bit_per_joule, dtype=float) - np.asarray(ee_mc.ee_bit_per_joule, dtype=float)
        summary["mean_abs_ee_gap_bit_per_joule"] = _safe_mean(np.abs(diff))

    return summary



# ------- Upstream prepration -------

def prepare_upstream_context(
        cfg: Optional[Config] = None,
        *,
        config_overrides: Optional[dict[str, Any]] = None,
        phase_mode: PhaseMode = "random",
        equal_phase: float = 0.0,
        seed_override: Optional[int] = None,        
) -> PreparedUpstream:
    """
    Prepare the upstream objects that are reusable across branches and many sweeps.

    This includes
    - geometry
    - large-scale fading
    - correlations
    - RIS phase shifts
    """
    base_cfg = make_default_config() if cfg is None else cfg
    cfg_eff = apply_config_overrides(base_cfg, config_overrides)

    base_seed = int(cfg_eff.sim.seed if seed_override is None else seed_override)
    seeds = _stage_seeds(base_seed=base_seed)

    geom = generate_geometry_setups(cfg_eff)
    ls = compute_large_scale_setups(cfg=cfg_eff, geom=geom, seed=seeds["large_scale"])
    corr = build_correlation_setups(cfg=cfg_eff, geom=geom, ls=ls, seed=seeds["large_scale"])

    phases = build_ris_phase_shifts_setups(
        cfg=cfg_eff,
        mode=phase_mode,
        seed=seeds["phase"],
        equal_phase=equal_phase,
    )

    return PreparedUpstream(
        cfg=cfg_eff,
        geom=geom,
        large_scale=ls,
        corr=corr,
        phases=phases,
    )



# ------- Single simulation run -------

def run_single_simulation(
        cfg: Optional[Config] = None,
        *,
        config_overrides: Optional[dict[str, Any]] = None,
        prepared_upstream: Optional[PreparedUpstream] = None,
        sinr_mode: SinrMode = "closed_form",
        serving_mode: ServingMode = "candidate",
        pilot_scheme: PilotScheme = "heuristic",
        phase_mode: PhaseMode = "random",
        power_scheme: PowerScheme = "equal",
        Q_neighbors: int = 3,
        protect_masters: bool = True,
        custom_serve_mask: Optional[np.ndarray] = None,
        custom_serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,
        equal_phase: float = 0.0,
        seed_override: Optional[int] = None,
        pilot_power_watt_per_user: Optional[np.ndarray] = None,
        tau_d_override: Optional[float] = None,
        channel_block_len: int = 128,
        estimation_block_len: int = 256,
        store_ris_links: bool = False,
        optimized_power_options: Optional[OptimizedPowerOptions] = None,
) -> SimulationResults:
    """
    Run one full experiment.

    Branch logic
    - closed_form: statistics-only estimation, no channels
    - monte_carlo: statistics + channels + u_hat
    - both: statistics once, then channels + u_hat once, and both SINR branches
    """
    if sinr_mode not in ("closed_form", "monte_carlo", "both"):
        raise ValueError("sinr_mode must be 'closed_form', 'monte_carlo', or 'both'.")
    
    base_cfg = make_default_config() if cfg is None else cfg
    cfg_eff = apply_config_overrides(base_cfg, config_overrides)

    base_seed = int(cfg_eff.sim.seed if seed_override is None else seed_override)
    seeds = _stage_seeds(base_seed)

    # Upstream prepration
    if prepared_upstream is None:
        upstream = prepare_upstream_context(
            cfg=cfg_eff,
            config_overrides=None,
            phase_mode=phase_mode,
            equal_phase=equal_phase,
            seed_override=seed_override,
        )
    else:
        if config_overrides is not None and _overrides_touch_upstream(config_overrides):
            raise ValueError(
                "prepared_upstream was provided, but config_overrides changes upstream parameters. "
                "Either remove those overrides or prepare a new upstream context."
            )
        # Reuse upstream but still allow downstream-only overrides on cfg
        upstream_cfg = prepared_upstream.cfg
        cfg_eff = apply_config_overrides(upstream_cfg, config_overrides)
        upstream = PreparedUpstream(
            cfg=cfg_eff,
            geom=prepared_upstream.geom,
            large_scale=prepared_upstream.large_scale,
            corr=prepared_upstream.corr,
            phases=prepared_upstream.phases,
        )

        # Pilot assignment
    pilot = assign_pilots_setups(
        cfg=upstream.cfg,
        geom=upstream.geom,
        ls=upstream.large_scale,
        corr=upstream.corr,
        phases=upstream.phases,
        scheme=pilot_scheme,
        Q=Q_neighbors,
        protect_masters=protect_masters,
        seed=seeds["pilot"],
    )

    # Serving set
        # Serving set
    serve_mask_used = _resolve_serving_mask(
        cfg=upstream.cfg,
        serving_mode=serving_mode,
        pilot_out=pilot,
        custom_serve_mask=custom_serve_mask,
        custom_serve_users=custom_serve_users,
    )

    # Estimation statistics are needed by the optimized closed-form power allocator,
    # so we build them before power allocation.
    est_stats = build_channel_estimation_statistics(
        cfg=upstream.cfg,
        corr=upstream.corr,
        ls=upstream.large_scale,
        pilot_of_user=pilot.pilot_of_user,
        pilot_groups=pilot.pilot_groups,
        phases=upstream.phases,
        pilot_power_watt_per_user=pilot_power_watt_per_user,
    )

    # Power allocation
    if power_scheme == "equal":
        power = allocate_downlink_power(
            upstream.cfg,
            scheme="equal",
            serve_mask=serve_mask_used,
        )
    elif power_scheme == "optimized":
        power = allocate_downlink_power(
            upstream.cfg,
            scheme="optimized",
            serve_mask=serve_mask_used,
            est=est_stats,
            corr=upstream.corr,
            ls=upstream.large_scale,
            pilot_of_user=pilot.pilot_of_user,
            phases=upstream.phases,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
            optimized_options=optimized_power_options,
        )
    else:
        raise ValueError(f"Unknown power_scheme {power_scheme}")

    channels: ChannelSetups | None = None
    est_full: ChannelEstimationSetups | None = None

    need_monte_carlo = sinr_mode in ("monte_carlo", "both")

    if need_monte_carlo:
        channels = generate_channels_setups(
            cfg=upstream.cfg,
            corr=upstream.corr,
            ls=upstream.large_scale,
            phases=upstream.phases,
            seed=seeds["channels"],
            block_len=channel_block_len,
            store_ris_links=store_ris_links,
        )

        est_full = estimate_u_mmse_realizations(
            cfg=upstream.cfg,
            channels=channels,
            est_stats=est_stats,
            pilot_groups=pilot.pilot_groups,
            seed=seeds["estimation"],
            block_len=estimation_block_len,
        )

    # SINR branches
    sinr_cf: DownlinkSinrClosedFormSetups | None = None
    sinr_mc: DownlinkSinrMonteCarloSetups | None = None

    se_cf: SpectralEfficiencySetups | None = None
    se_mc: SpectralEfficiencySetups | None = None

    ee_cf: EnergyEfficiencySetups | None = None
    ee_mc: EnergyEfficiencySetups | None = None

    if sinr_mode in ("closed_form", "both"):
        sinr_cf = compute_dl_sinr_closed_form(
            cfg=upstream.cfg,
            est=est_stats,
            corr=upstream.corr,
            ls=upstream.large_scale,
            power=power,
            pilot_of_user=pilot.pilot_of_user,
            phases=upstream.phases,
            serve_mask=serve_mask_used,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
        )

        se_cf = compute_spectral_efficiency_setups(
            upstream.cfg,
            sinr_cf.sinr_dl,
            tau_d=tau_d_override,
        )

        ee_cf = compute_energy_efficiency_setups(
            upstream.cfg,
            se_cf,
            power,
            serve_mask=serve_mask_used,
        )

    if sinr_mode in ("monte_carlo", "both"):
        if channels is None or est_full is None:
            raise RuntimeError("Monte Carlo branch requested, but channels or full estimation is missing")

        sinr_mc = compute_dl_sinr_monte_carlo(
            cfg=upstream.cfg,
            channels=channels,
            est=est_full,
            power=power,
            pilot_of_user=pilot.pilot_of_user,
            serve_mask=serve_mask_used,
        )

        se_mc = compute_spectral_efficiency_setups(
            upstream.cfg,
            sinr_mc.sinr_dl,
            tau_d=tau_d_override,
        )

        ee_mc = compute_energy_efficiency_setups(
            upstream.cfg,
            se_mc,
            power,
            serve_mask=serve_mask_used,
        )

    summary = _build_summary(
        serve_mask_used=serve_mask_used,
        sinr_cf=sinr_cf,
        sinr_mc=sinr_mc,
        se_cf=se_cf,
        se_mc=se_mc,
        ee_cf=ee_cf,
        ee_mc=ee_mc,
    )

    return SimulationResults(
        cfg=upstream.cfg,
        prepared_upstream=upstream,
        pilot=pilot,
        serve_mask_used=serve_mask_used,
        power=power,
        channels=channels,
        est_stats=est_stats,
        est_full=est_full,
        sinr_closed_form=sinr_cf,
        sinr_monte_carlo=sinr_mc,
        se_closed_form=se_cf,
        se_monte_carlo=se_mc,
        ee_closed_form=ee_cf,
        ee_monte_carlo=ee_mc,
        summary=summary,
    )



# ------- Sweep runner -------

def run_parameter_sweep(
    sweep_name: str,
    sweep_values: Sequence[Any],
    cfg: Optional[Config] = None,
    *,
    base_overrides: Optional[dict[str, Any]] = None,
    sinr_mode: SinrMode = "closed_form",
    serving_mode: ServingMode = "candidate",
    pilot_scheme: PilotScheme = "heuristic",
    phase_mode: PhaseMode = "random",
    power_scheme: PowerScheme = "equal",
    Q_neighbors: int = 3,
    protect_masters: bool = True,
    custom_serve_mask: Optional[np.ndarray] = None,
    custom_serve_users: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    equal_phase: float = 0.0,
    seed_override: Optional[int] = None,
    pilot_power_watt_per_user: Optional[np.ndarray] = None,
    tau_d_override: Optional[float] = None,
    channel_block_len: int = 128,
    estimation_block_len: int = 256,
    store_ris_links: bool = False,
    reuse_upstream: bool = False,
    optimized_power_options: Optional[OptimizedPowerOptions] = None,
) -> SweepResults:
    """
    Sweep one config parameter.

    Example
    - sweep_name = "pilots.pilot_len"
    - sweep_values = [4, 6, 8, 10]

    If reuse_upstream is True, the same upstream objects are reused across sweep points.
    This is ideal for downstream sweeps such as tau_p, power budgets, or serving-policy comparisons.

    Important
    - If the sweep or base overrides touch upstream parameters, reuse_upstream is not allowed.
    """
    base_cfg = make_default_config() if cfg is None else cfg
    base_overrides_eff = {} if base_overrides is None else dict(base_overrides)

    if reuse_upstream:
        if _overrides_touch_upstream(base_overrides_eff):
            raise ValueError("base_overrides changes upstream parameters, so reuse_upstream=True is not valid")

        if _overrides_touch_upstream({sweep_name: sweep_values[0]}):
            raise ValueError(
                f"Sweep variable '{sweep_name}' changes upstream objects, so reuse_upstream=True is not valid"
            )

        prepared = prepare_upstream_context(
            cfg=apply_config_overrides(base_cfg, base_overrides_eff),
            config_overrides=None,
            phase_mode=phase_mode,
            equal_phase=equal_phase,
            seed_override=seed_override,
        )
    else:
        prepared = None

    results: list[SimulationResults] = []
    summary_rows: list[dict[str, float | int | None]] = []

    for value in sweep_values:
        overrides = dict(base_overrides_eff)
        overrides[sweep_name] = value

        result = run_single_simulation(
            cfg=base_cfg,
            config_overrides=overrides,
            prepared_upstream=prepared,
            sinr_mode=sinr_mode,
            serving_mode=serving_mode,
            pilot_scheme=pilot_scheme,
            phase_mode=phase_mode,
            power_scheme=power_scheme,
            Q_neighbors=Q_neighbors,
            protect_masters=protect_masters,
            custom_serve_mask=custom_serve_mask,
            custom_serve_users=custom_serve_users,
            equal_phase=equal_phase,
            seed_override=seed_override,
            pilot_power_watt_per_user=pilot_power_watt_per_user,
            tau_d_override=tau_d_override,
            channel_block_len=channel_block_len,
            estimation_block_len=estimation_block_len,
            store_ris_links=store_ris_links,
            optimized_power_options=optimized_power_options,
        )

        row = dict(result.summary)
        row["sweep_value"] = value

        results.append(result)
        summary_rows.append(row)

    return SweepResults(
        sweep_name=sweep_name,
        sweep_values=list(sweep_values),
        results=results,
        summary_rows=summary_rows,
    )