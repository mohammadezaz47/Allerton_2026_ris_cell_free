# ITNG YMYFA 80asra YA
# IMZZ

# src/config.py

"""
config.py

Central place for simulation parameters.

Goal
- Every constant that defines the simulation lives here
- Results remain reproducible months later
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Tuple



#------- Small helpers -------

def dbm_to_watt(p_dbm: float) -> float:
    """Convert power in dBm to Watt"""
    return 10 ** ((p_dbm - 30.0) / 10.0)



def watt_to_dbm(p_watt: float) -> float:
    """Convert power in Watt to dBm"""
    if p_watt <= 0:
        raise ValueError("Power in Watt must be positive to convert to dBm")
    return 10.0 * math.log10(p_watt) + 30.0



#------- Config blocks -------

@dataclass(frozen=True)
class DimensionsConfig:
    """System dimensions"""
    num_users: int = 10             # K
    num_aps: int = 35               # L
    num_ap_antennas: int = 4        # M per AP

    # Single RIS for now
    num_ris: int = 1

    # RIS layout. Total elements N = ris_n_hor * ris_n_ver
    ris_n_hor: int = 10
    ris_n_ver: int = 10

    @property
    def num_ris_elements(self) -> int:
        return self.ris_n_hor * self.ris_n_ver
    

@dataclass(frozen=True)
class NoiseConfig:
    """
    Noise and bandwidth.
    
    noise(dBm) = thermal_noise_dbm_per_hz + 10 * log10(B) + NF
    """
    bandwidth_hz: float = 1e6
    noise_figure_db: float = 7.0
    thermal_noise_dbm_per_hz: float = -174.0

    @property
    def noise_power_dbm(self) -> float:
        return (
            self.thermal_noise_dbm_per_hz
            + 10.0 * math.log10(self.bandwidth_hz)
            + self.noise_figure_db
        )
    
@dataclass(frozen=True)
class PilotConfig:
    """Coherence and pilots."""
    coherence_block_length: float = 200.0    # tau_c
    pilot_power_watt: float = 0.1            # ulpink pilot power per user

    # if None we will default to tau_p = K
    pilot_len: float | None = None

    def pilot_len_effective(self, num_users: int) -> float:
        return self.pilot_len if self.pilot_len is not None else float(num_users)
    


@dataclass(frozen=True)
class ArrayConfig:
    """Array and correlation knob."""
    ap_spacing_wavelength: float = 0.5
    ris_spacing_wavelength: float = 0.5

    asd_azim_deg: float = 15.0
    asd_elev_deg: float = 15.0

    ap_scattering_distribution: str = "gaussian"
    ap_scattering_integration_points: int = 4001



@dataclass(frozen=True)
class HeightConfig:
    """Heights for 3D distance per link"""
    ap_height_m: float = 15.0
    user_height_m: float = 1.65
    ris_height_m: float = 30.0



@dataclass(frozen=True)
class PropagationConfig:
    """
    Large-scale fading.

    LoS gain in dB
      pl_los_intercept_db + pl_los_slope_db_per_log10m * log10(d)

    NLoS gain in dB
      pl_nlos_intercept_db + pl_nlos_slope_db_per_log10m * log10(d)
    """
    shadow_std_los_db: float = 4.0
    shadow_std_nlos_db: float = 10.0

    min_distance_m: float = 20.0
    max_los_distance_m: float = 300.0

    pl_los_intercept_db: float = -30.18
    pl_los_slope_db_per_log10m: float = -26.0

    pl_nlos_intercept_db: float = -34.53
    pl_nlos_slope_db_per_log10m: float = -38.0 - 4.0

    direct_loss_db: float = 10.0

    ap_ris_condition: Literal["los", "nlos", "probabilistic"] = "los"
    ris_user_condition: Literal["los", "nlos", "probabilistic"] = "los"
    ap_user_condition: Literal["los", "nlos", "probabilistic"] = "nlos"

    ap_user_block_prob: float = 0.0
    blockage_mode: Literal["zero", "extra_loss_db"] = "zero"
    blockage_extra_loss_db: float = 80.0

    use_rician_k_factor: bool = False
    k_factor_intercept_db: float = 13.0
    k_factor_slope_db_per_m: float = -0.03



@dataclass(frozen=True)
class FadingConfig:
    """Small-scale fading model"""
    small_scale: Literal["rayleigh", "rician"] = "rayleigh"



@dataclass(frozen=True)
class GeometryConfig:
    """
    Geometry in 2D meters.

    Square area contains a circular hotspot.
    UEs are dropped in the hotspot.
    APs are dropped in the square outside hotspot plus guard zone.
    RIS can be at hotspot center or on hotspot edge.
    """
    area_center_xy: Tuple[float, float] = (0.0, 0.0)
    area_side_m: float = 1000.0

    hotspot_center_xy: Tuple[float, float] = (220.0, 0.0)
    hotspot_radius_m: float = 60.0

    guard_width_m: float = 140.0

    # Prevent users to close to the RIS
    ris_user_exclusion_radius_m: float = 15.0

    # AP placement policy
    ap_placement: Literal["random", "grid"] = "random"

    # Random AP drop controls
    ap_min_seperation_m: float = 8.0
    ap_drop_max_tries: int = 200000

    # Grid AP drop controls
    grid_spacing_m: float = 50.0
    grid_offset_xy: Tuple[float, float] = (0.0, 0.0)

    # RIS placement
    ris_placement: Literal["center", "edge"] = "edge"
    ris_edge_angle_deg: float = 180.0
    ris_edge_inset_m: float = 3.0

    # Wrap-around toggle for boundary effects
    wraparound_enabled: bool = False

    ap_region_mode: Literal["full_area", "rectangle"] = "rectangle"
    ap_region_bounds_xyxy: Tuple[float, float, float, float] = (-500.0, -120.0, -250.0, 250.0)



@dataclass(frozen=True)
class UserCentricConfig:
    """
    User-centric association knobs.

    Association logic lives elsewhere.
    We store the policy knobs here for reproducibility.
    """
    kappa: float = 0.5

    max_aps_per_user: int | None = None
    min_aps_per_user: int = 1

    # If None, treat it as tau_p in the association module
    max_users_per_ap: int | None = None



@dataclass(frozen=True)
class RISConfig:
    """RIS toggles."""
    enable_ris: bool = True



@dataclass(frozen=True)
class PowerConfig:
    """Transmit power constraints used in UL and DL."""
    ul_max_power_watt_per_user: float = 0.2
    dl_max_power_watt_per_ap: float = 1.0



@dataclass(frozen=True)
class EnergyEfficiencyConfig:
    """
    Energy efficiency parameters.

    Units
    - pa_efficiency: unitless
    - ap_circuit_power_watt: Watt
    - ris_static_power_watt: Watt
    - fronthaul_energy_per_bit_joule: Joule/bit
    """
    pa_efficiency: float = 0.39
    ap_circuit_power_watt: float = 0.2
    ris_static_power_watt: float = 0.5
    fronthaul_energy_per_bit_joule: float = 1e-12

    # If True, RIS static power is counted only when RIS is enabled
    use_ris_power_only_when_enabled: bool = True



@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo controls."""
    num_setups: int = 10
    num_realizations: int = 10000
    seed: int = 1



@dataclass(frozen=True)
class Config:
    """Top-level config bundle."""
    dims: DimensionsConfig = DimensionsConfig()
    noise: NoiseConfig = NoiseConfig()
    pilots: PilotConfig = PilotConfig()
    arrays: ArrayConfig = ArrayConfig()
    heights: HeightConfig = HeightConfig()
    prop: PropagationConfig = PropagationConfig()
    fading: FadingConfig = FadingConfig()
    geom: GeometryConfig = GeometryConfig()
    uc: UserCentricConfig = UserCentricConfig()
    ris: RISConfig = RISConfig()
    power: PowerConfig = PowerConfig()
    ee: EnergyEfficiencyConfig = EnergyEfficiencyConfig()
    sim: SimulationConfig = SimulationConfig()

    # Small derived helpers that other modules can call
    def tau_p(self) -> int:
        return int(round(self.pilots.pilot_len_effective(self.dims.num_users)))
    

    def max_users_per_ap_effective(self) -> int:
        return self.uc.max_users_per_ap if self.uc.max_users_per_ap is not None else self.tau_p()
    

    def validate(self) -> None:
        if self.dims.num_users <= 0:
            raise ValueError("num_users must be positive")
        if self.dims.num_aps <= 0:
            raise ValueError("num_aps must be positive")
        if self.dims.num_ap_antennas <= 0:
            raise ValueError("num_ap_antennas must be positive")

        if self.dims.num_ris < 0:
            raise ValueError("num_ris must be at least 0")
        if self.dims.num_ris > 0 and (self.dims.ris_n_hor <= 0 or self.dims.ris_n_ver <= 0):
            raise ValueError("RIS dimensions must be positive when num_ris > 0")

        if self.noise.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")

        if self.prop.min_distance_m <= 0:
            raise ValueError("min_distance_m must be positive")

        if self.geom.area_side_m <= 0:
            raise ValueError("area_side_m must be positive")
        if self.geom.hotspot_radius_m <= 0:
            raise ValueError("hotspot_radius_m must be positive")
        if self.geom.guard_width_m < 0:
            raise ValueError("guard_width_m must be non-negative")
        if self.geom.ris_user_exclusion_radius_m < 0:
            raise ValueError("ris_user_exclusion_radius_m must be non-negative")

        if self.geom.ap_drop_max_tries <= 0:
            raise ValueError("ap_drop_max_tries must be positive")
        if self.geom.grid_spacing_m <= 0:
            raise ValueError("grid_spacing_m must be positive")

        if self.uc.min_aps_per_user <= 0:
            raise ValueError("min_aps_per_user must be positive")
        if self.uc.max_aps_per_user is not None and self.uc.max_aps_per_user < self.uc.min_aps_per_user:
            raise ValueError("max_aps_per_user must be >= min_aps_per_user")

        if self.power.ul_max_power_watt_per_user < 0:
            raise ValueError("ul_max_power_watt_per_user must be non-negative")
        if self.power.dl_max_power_watt_per_ap < 0:
            raise ValueError("dl_max_power_watt_per_ap must be non-negative")

        if not (0 < self.ee.pa_efficiency <= 1):
            raise ValueError("pa_efficiency must be in the interval (0, 1]")
        if self.ee.ap_circuit_power_watt < 0:
            raise ValueError("ap_circuit_power_watt must be non-negative")
        if self.ee.ris_static_power_watt < 0:
            raise ValueError("ris_static_power_watt must be non-negative")
        if self.ee.fronthaul_energy_per_bit_joule < 0:
            raise ValueError("fronthaul_energy_per_bit_joule must be non-negative")
        
        if self.sim.num_setups <= 0:
            raise ValueError("num_setups must be positive")
        if self.sim.num_realizations <= 0:
            raise ValueError("num_realizations must be positive")
        
        tau_p = self.tau_p()
        tau_c = int(round(self.pilots.coherence_block_length))

        if tau_p <= 0:
            raise ValueError("tau_p must be positive")
        if tau_p > tau_c:
            raise ValueError("tau_p cannot be larger than tau_c")
        if self.pilots.pilot_power_watt < 0:
            raise ValueError("pilot_power_watt must be non-negative")
        


def make_default_config() -> Config:
    """Factory for a validated default config."""
    cfg = Config()
    cfg.validate()
    return cfg