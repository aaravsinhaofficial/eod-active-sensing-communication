"""Physical and morphological constants.

All values are transcribed from the upstream WEF simulator
(Singh, Johnson-Yu et al. 2025, https://github.com/KempnerInstitute/wef),
file `onpolicy/custom/fish/cfg.py`, so that the GPU reimplementation in this
package is numerically comparable to the reference NumPy implementation.

Distances are in centimetres unless a name ends in `_m`.
"""

import math

import numpy as np

# --------------------------------------------------------------------------
# Universal constants
# --------------------------------------------------------------------------
K_COULOMB = 8.99e9  # N m^2 / C^2
EPSILON_0 = 8.854e-12  # F / m
EPS_M = 1e-5  # m, softening added to all source-sensor distances
CM_TO_M = 1e-2
M_TO_CM = 1e2
MVCM_TO_VM = 0.1  # mV/cm -> V/m

# --------------------------------------------------------------------------
# Electric organ / body geometry
# --------------------------------------------------------------------------
MONOPOLE_POSITIONS_EGO = np.array([[0.5, 0.0], [-0.5, 0.0]])  # cm, head/tail poles
MONOPOLE_CHARGES = np.array([1.0, -1.0]) * 1.11e-15  # C  (Chen et al. 2005)
FISH_INTRINSIC_DIPOLE_MOMENT = np.array([1.11e-23, 0.0])  # C m, ego frame
FOOD_INTRINSIC_DIPOLE_MOMENT = np.array([0.0, 1.11e-24])  # C m, ego frame
# A predator is a large fish: same kind of standing bioelectric field, stronger,
# so it becomes passively detectable at roughly 25 cm rather than 8 cm.  Prey
# still hear the predator far later than the predator hears their discharges.
PRED_INTRINSIC_DIPOLE_MOMENT = np.array([3.0e-22, 0.0])  # C m, ego frame

BODY_RADIUS_CM = 1.0
FOOD_RADIUS_CM = 0.25
FISH_CONTRAST = -0.5  # insulating body, Chen et al. eq. 6
FOOD_CONTRAST = -0.5

# --------------------------------------------------------------------------
# Electroreceptor arrays
# --------------------------------------------------------------------------
NUM_MORM = 36  # mormyromasts: active electrolocation (self-EOD reafference)
NUM_MORM_CHIN = int(NUM_MORM * 0.3)  # 10, densely packed on the Schnauzenorgan
NUM_MORM_REST = NUM_MORM - NUM_MORM_CHIN  # 26
NUM_KNOLLEN = 12  # knollenorgans: conspecific-EOD detection (the social channel)
NUM_AMP = 24  # ampullary organs: passive low-frequency (DC) sensing
CHIN_ANGLE = math.pi / 3

# Sensor dynamic ranges (V/m)
GENERAL_SENSOR_MIN = 1e-25
AMP_SENSOR_MIN = 2e-9 * MVCM_TO_VM
AMP_SENSOR_MAX = 2e-7 * MVCM_TO_VM
KNOLLEN_SENSOR_MIN = 2e-6 * MVCM_TO_VM
KNOLLEN_BINARIZE_THRESHOLD = KNOLLEN_SENSOR_MIN
MAX_CHARGE_ALLOWED = MONOPOLE_CHARGES[0]

# Mormyromast dynamic range (upstream SENSING_PARAMS_DYNAMIC, the configuration
# used for the published foraging runs).
MORM_SENSOR_MIN = 2 * 0.25 * 1e-6 * MVCM_TO_VM  # 5.0e-8 V/m
MORM_SENSOR_MAX = 2 * 0.25 * MVCM_TO_VM  # 5.0e-2 V/m
# Object images carried by a *conspecific's* discharge are far weaker than those
# carried by one's own; upstream rescales them so they fall inside the receptor
# range (cfg.SENSING_PARAMS_DYNAMIC["mormyromast_sensor_cons_multiplier"]).
MORM_CONS_MULTIPLIER = 100.0

# --------------------------------------------------------------------------
# Walls
# --------------------------------------------------------------------------
DO_WALL_REFLECTION = True
# NB: upstream `cfg.ELECTRIC_CONSTANTS["reflection_scale"]` is 0.95, but
# `ElectricScene._build_reflections` calls `reflect_sources` without forwarding
# it, so the sensing path actually runs with unattenuated first-order images.
# We match the code that produced the published results, not the unused config
# value.
REFLECTION_SCALE = 1.0
FLIP_ON_REFLECTION = False  # non-conducting walls -> no charge flip

# --------------------------------------------------------------------------
# Locomotion (90th percentile of real-fish motion statistics; 95th for angular)
# --------------------------------------------------------------------------
FPS_SIM = 83.0
DT = 1.0 / FPS_SIM  # s

MAX_LINEAR_VELOCITY = 0.35 * M_TO_CM / FPS_SIM  # cm / step
MIN_LINEAR_VELOCITY = -0.05 * M_TO_CM / FPS_SIM
MAX_ANGULAR_VELOCITY = 3.6 / FPS_SIM  # rad / step
MIN_ANGULAR_VELOCITY = -3.5 / FPS_SIM

# --------------------------------------------------------------------------
# Task / reward (upstream REWARDS dict).  Note that the upstream reward has no
# EOD-emission term: pulses are free.  This package adds `eod_cost`, which is
# zero by default so that the baseline condition reproduces upstream.
# --------------------------------------------------------------------------
REWARDS = {
    "timestep": -0.0,
    "eat": 10.0,
    "proximity_shaping": 1.0,
    "bitten": -5.0,
    "bite": -0.001,
    "collision": -0.5,
    "effort_over": -0.1,
    "eod_cost": 0.0,  # NEW: metabolic cost per discharge
    "predation": -0.0,  # NEW: cost of being detected/struck by a predator
}

EATING_RADIUS_CM = 2.0
EATING_ANGLE = math.pi / 4
BITING_RADIUS_CM = 3.0
EAT_COOLDOWN_STEPS = 3
BITE_COOLDOWN_STEPS = 5


def mormyromast_angles() -> np.ndarray:
    """Mormyromast placement: dense on the chin, sparse elsewhere.

    Mirrors upstream `calculate_mormyromast_angles` with `asym_rays=True`.
    """
    chin = np.linspace(-CHIN_ANGLE / 2, CHIN_ANGLE / 2, NUM_MORM_CHIN)
    remaining = 2 * math.pi - CHIN_ANGLE
    rest = np.linspace(
        CHIN_ANGLE / 2, CHIN_ANGLE / 2 + remaining, NUM_MORM_REST + 1
    )[:-1]
    return np.concatenate([chin, rest])


def uniform_angles(n: int) -> np.ndarray:
    return np.linspace(0.0, 2 * math.pi, n, endpoint=False)
