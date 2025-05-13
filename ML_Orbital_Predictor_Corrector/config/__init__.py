# my_ai_orbital_project/config/__init__.py
"""
Configuration Package for the AI Orbital Project.
This __init__.py makes it easier to access configuration variables
from app_config, sim_params, and model_hyperparams.
"""
print("Initializing master config package...")

# Expose variables from app_config.py
from .app_config import (
    MODEL_ARTIFACTS_DIR,
    RAW_GENERATED_DATA_DIR,
    NORMALIZED_GENERATED_DATA_DIR,
    SIMULATION_OUTPUT_DIR
)

# Expose variables from sim_params.py
from .sim_params import (
    TIME_STEP_S,
    MAX_SIM_TIME_S,
    # NUM_ORBITS_TO_SIMULATE,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    J2_EARTH,
    ENABLE_J2_PERTURBATION,
    ENABLE_DRAG_PERTURBATION,
    SPACECRAFT_MASS_KG,
    SPACECRAFT_DRAG_COEFF,
    SPACECRAFT_DRAG_AREA_M2,
    # ATMOSPHERIC_DENSITY_MODEL_TYPE,
    ODE_SOLVER_METHOD,
    ODE_RELATIVE_TOLERANCE,
    ODE_ABSOLUTE_TOLERANCE,
    DEVIATION_THRESHOLD_KM,
    PREDICTION_HORIZON_S,
    MANEUVER_EXECUTION_ASSUMPTION,
    REF_ORBIT_SEMI_MAJOR_AXIS_KM,
    REF_ORBIT_ECCENTRICITY,
    REF_ORBIT_INCLINATION_DEG,
    REF_ORBIT_RAAN_DEG,
    REF_ORBIT_ARG_PERIGEE_DEG,
    REF_ORBIT_TRUE_ANOMALY_DEG, OMEGA_EARTH
)

# Expose dictionaries/variables from model_hyperparams.py
from .model_hyperparams import (
    VALIDATION_SPLIT_RATIO,
    RANDOM_SEED,
    AI1_MODEL_CONFIG,
    AI2_MODEL_CONFIG
)

# Define __all__ to specify what 'from config import *' would import
__all__ = [
    # From app_config
    'MODEL_ARTIFACTS_DIR', 'RAW_GENERATED_DATA_DIR', 'NORMALIZED_GENERATED_DATA_DIR',
    'SIMULATION_OUTPUT_DIR',
    # From sim_params
    'TIME_STEP_S', 'MAX_SIM_TIME_S', 'MU_EARTH_KM3_S2', 'R_EARTH_KM', 'J2_EARTH','OMEGA_EARTH',
    'ENABLE_J2_PERTURBATION', 'ENABLE_DRAG_PERTURBATION', 'SPACECRAFT_MASS_KG',
    'SPACECRAFT_DRAG_COEFF', 'SPACECRAFT_DRAG_AREA_M2', 'ODE_SOLVER_METHOD',
    'ODE_RELATIVE_TOLERANCE', 'ODE_ABSOLUTE_TOLERANCE', 'DEVIATION_THRESHOLD_KM',
    'PREDICTION_HORIZON_S', 'MANEUVER_EXECUTION_ASSUMPTION',
    'REF_ORBIT_SEMI_MAJOR_AXIS_KM', 'REF_ORBIT_ECCENTRICITY', 'REF_ORBIT_INCLINATION_DEG',
    'REF_ORBIT_RAAN_DEG', 'REF_ORBIT_ARG_PERIGEE_DEG', 'REF_ORBIT_TRUE_ANOMALY_DEG',
    # From model_hyperparams
    'VALIDATION_SPLIT_RATIO', 'RANDOM_SEED', 'AI1_MODEL_CONFIG', 'AI2_MODEL_CONFIG'
]

print("Master config package initialized successfully.")